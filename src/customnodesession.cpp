//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "customnodesession.hpp"

#include <cstdint>
#include <functional>
#include <utility>

#include "custom_node_output_allocator.hpp"
#include "logging.hpp"
#include "node.hpp"
#include "node_library.hpp"
#include "nodeinputhandler.hpp"
#include "pipelineeventqueue.hpp"

namespace ovms {

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const NodeLibrary& library, session_id_t shardsCount) :
    NodeSession(metadata, nodeName, inputsCount, shardsCount),
    library(library) {}

CustomNodeSession::CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const NodeLibrary& library, session_id_t shardsCount) :
    NodeSession(std::move(metadata), nodeName, inputsCount, shardsCount),
    library(library) {}

CustomNodeSession::~CustomNodeSession() = default;

Status CustomNodeSession::execute(PipelineEventQueue& notifyEndQueue, Node& node, const NodeLibrary& library, std::unique_ptr<struct CustomNodeParam[]>& parameters, int parametersLength) {
    const auto& blobMap = this->inputHandler->getInputs();
    auto inputTensorsLength = blobMap.size();
    auto inputTensors = std::make_unique<struct CustomNodeTensor[]>(inputTensorsLength);
    struct CustomNodeTensor* outputTensors = nullptr;
    int outputTensorsLength = 0;

    int i = 0;
    for (const auto& [name, blob] : blobMap) {
        inputTensors[i].name = static_cast<const char*>(name.c_str());
        inputTensors[i].data = static_cast<uint8_t*>(blob->buffer());
        inputTensors[i].dataLength = static_cast<uint64_t>(blob->byteSize());
        inputTensors[i].dims = static_cast<uint64_t*>(blob->getTensorDesc().getDims().data());
        inputTensors[i].dimsLength = static_cast<uint64_t>(blob->getTensorDesc().getDims().size());
        inputTensors[i].precision = toCustomNodeTensorPrecision(blob->getTensorDesc().getPrecision());
        i++;
    }

    int result = library.execute(
        inputTensors.get(),
        inputTensorsLength,
        &outputTensors,
        &outputTensorsLength,
        parameters.get(),
        parametersLength);

    // If result is not 0, it means execution has failed.
    // In this case shared library is responsible for cleaning up resources (memory).
    if (result != 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has failed custom node execution with return code: {}", getName(), getSessionKey(), result);
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_EXECUTION_FAILED;
    }

    if (outputTensors == nullptr) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has corrupted outputs handle", getName(), getSessionKey());
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED;
    }

    if (outputTensorsLength <= 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has corrupted number of outputs", getName(), getSessionKey());
        // Cleanup whatever is possible
        this->library.releaseTensors(outputTensors);
        notifyEndQueue.push({node, getSessionKey()});
        return StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT;
    }

    // At this point this is important we do not exit before finishing the loop.
    // There will be memory leak if any tensor is not converted into blob.
    // Blob destructor is responsible for cleaning up resources.
    Status status = StatusCode::OK;
    for (int i = 0; i < outputTensorsLength; i++) {
        InferenceEngine::Blob::Ptr resultBlob;
        auto result = this->createBlob(&outputTensors[i], resultBlob);
        if (!result.ok()) {
            SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; has corrupted output: {}; cannot convert to blob", getName(), getSessionKey(), outputTensors[i].name);
            // Cleanup whatever is possible
            this->library.releaseBuffer(&outputTensors[i]);
            if (status.ok()) {
                status = result;
            }
            continue;
        }
        this->resultBlobs.emplace(std::make_pair(std::string(outputTensors[i].name), std::move(resultBlob)));
    }

    this->library.releaseTensors(outputTensors);

    notifyEndQueue.push({node, getSessionKey()});
    return status;
}

Status CustomNodeSession::fetchResult(const std::string& name, InferenceEngine::Blob::Ptr& resultBlob) {
    auto it = resultBlobs.find(name);
    if (it == resultBlobs.end()) {
        return StatusCode::NODE_LIBRARY_MISSING_OUTPUT;
    }
    resultBlob = it->second;
    return StatusCode::OK;
}

Status CustomNodeSession::createBlob(const struct CustomNodeTensor* tensor, InferenceEngine::Blob::Ptr& resultBlob) {
    InferenceEngine::TensorDesc desc;

    // precision
    InferenceEngine::Precision precision = toInferenceEnginePrecision(tensor->precision);
    if (precision == InferenceEngine::Precision::UNSPECIFIED) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; Unspecified output precision from custom node",
            this->getName(),
            this->getSessionKey());
        return StatusCode::INVALID_PRECISION;
    }
    desc.setPrecision(precision);

    // shape
    if (tensor->dims == nullptr || tensor->dimsLength == 0) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "CNode {}; session: {}; Corrupted output shape",
            this->getName(),
            this->getSessionKey());
        return StatusCode::INVALID_SHAPE;
    }
    InferenceEngine::SizeVector shape(tensor->dims, tensor->dims + tensor->dimsLength);
    desc.setDims(shape);

    // data
    size_t expectedElementsCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_t>());
    size_t expectedDataLength = expectedElementsCount *= precision.size();
    if (tensor->data == nullptr || tensor->dataLength != expectedDataLength) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Node {}; session: {}; Corrupted output data",
            this->getName(),
            this->getSessionKey());
        return StatusCode::INVALID_CONTENT_SIZE;
    }
    auto allocator = std::make_shared<CustomNodeOutputAllocator>(*tensor, this->library);
    try {
        switch (tensor->precision) {
        case CustomNodeTensorPrecision::FP32:
            resultBlob = InferenceEngine::make_shared_blob<float>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::FP16:
            resultBlob = InferenceEngine::make_shared_blob<uint16_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I16:
            resultBlob = InferenceEngine::make_shared_blob<int16_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::U16:
            resultBlob = InferenceEngine::make_shared_blob<uint16_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I32:
            resultBlob = InferenceEngine::make_shared_blob<int32_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::I8:
            resultBlob = InferenceEngine::make_shared_blob<int8_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::U8:
            resultBlob = InferenceEngine::make_shared_blob<uint8_t>(desc, std::move(allocator));
            break;
        case CustomNodeTensorPrecision::UNSPECIFIED:
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "{}: {}", status.string(), e.what());
        return status;
    } catch (std::logic_error& e) {
        Status status = StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR;
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "{}: {}", status.string(), e.what());
        return status;
    }

    resultBlob->allocate();

    return StatusCode::OK;
}

void CustomNodeSession::clearInputs() {
    this->inputHandler->clearInputs();
}

void CustomNodeSession::release() {
}

}  // namespace ovms
