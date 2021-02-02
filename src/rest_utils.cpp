//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "rest_utils.hpp"

#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/util/json_tensor.h"
#pragma GCC diagnostic pop

#define DEBUG
#include "timer.hpp"

using tensorflow::DataType;
using tensorflow::DataTypeSize;
using tensorflow::serving::JsonPredictRequestFormat;
using tensorflow::serving::MakeJsonFromTensors;
using tensorflow::serving::PredictResponse;

namespace ovms {

Status checkValField(const size_t& fieldSize, const size_t& expectedElementsNumber) {
    if (fieldSize == 0)
            return StatusCode::REST_SERIALIZE_NO_DATA;
    if (fieldSize != expectedElementsNumber)
            return StatusCode::REST_SERIALIZE_VAL_FIELD_INVALID_SIZE;
    return StatusCode::OK;
}

Status makeJsonFromPredictResponse(
    PredictResponse& response_proto,
    std::string* response_json,
    Order order) {
    if (order == Order::UNKNOWN) {
        return StatusCode::REST_PREDICT_UNKNOWN_ORDER;
    }

    Timer timer;
    using std::chrono::microseconds;

    timer.start("convert");

    for (auto& kv : *response_proto.mutable_outputs()) {
        auto& tensor = kv.second;

        size_t expectedContentSize = DataTypeSize(tensor.dtype());
        for (int i = 0; i < tensor.tensor_shape().dim_size(); i++) {
            expectedContentSize *= tensor.tensor_shape().dim(i).size();
        }
        size_t expectedElementsNumber = expectedContentSize / DataTypeSize(tensor.dtype());
        bool seekDataInValField = false;

        if (tensor.tensor_content().size() == 0)
            seekDataInValField = true;
        else if (tensor.tensor_content().size() != expectedContentSize) 
            return StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE;
        

        switch (tensor.dtype()) {
        case DataType::DT_FLOAT:
            if (seekDataInValField) {
                auto status = checkValField(tensor.float_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(float))
                    tensor.add_float_val(*reinterpret_cast<float*>(tensor.mutable_tensor_content()->data() + i));     
            }
            break;
        case DataType::DT_DOUBLE:
            if (seekDataInValField) {
                    auto status = checkValField(tensor.double_val_size(), expectedElementsNumber);
                    if (!status.ok())
                        return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(double))
                    tensor.add_double_val(*reinterpret_cast<double*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT32:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int32_t))
                    tensor.add_int_val(*reinterpret_cast<int32_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT16:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int16_t))
                    tensor.add_int_val(*reinterpret_cast<int16_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT8:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int8_t))
                    tensor.add_int_val(*reinterpret_cast<int8_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_UINT8:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(uint8_t))
                    tensor.add_int_val(*reinterpret_cast<uint8_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_INT64:
            if (seekDataInValField) {
                auto status = checkValField(tensor.int64_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(int64_t))
                    tensor.add_int64_val(*reinterpret_cast<int64_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_UINT32:
            if (seekDataInValField) {
                auto status = checkValField(tensor.uint32_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(uint32_t))
                    tensor.add_uint32_val(*reinterpret_cast<uint32_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        case DataType::DT_UINT64:
            if (seekDataInValField) {
                auto status = checkValField(tensor.uint64_val_size(), expectedElementsNumber);
                if (!status.ok())
                    return status;
            } else {
                for (size_t i = 0; i < tensor.tensor_content().size(); i += sizeof(uint64_t))
                    tensor.add_uint64_val(*reinterpret_cast<uint64_t*>(tensor.mutable_tensor_content()->data() + i));
            }
            break;
        default:
            return StatusCode::REST_UNSUPPORTED_PRECISION;
        }
    }

    timer.stop("convert");
    timer.start("MakeJsonFromTensors");

    const auto& tf_status = MakeJsonFromTensors(
        response_proto.outputs(),
        order == Order::ROW ? JsonPredictRequestFormat::kRow : JsonPredictRequestFormat::kColumnar,
        response_json);

    timer.stop("MakeJsonFromTensors");
    SPDLOG_DEBUG("tensor_content to *_val container conversion: {:.3f} ms", timer.elapsed<microseconds>("convert") / 1000);
    SPDLOG_DEBUG("MakeJsonFromTensors call: {:.3f} ms", timer.elapsed<microseconds>("MakeJsonFromTensors") / 1000);

    if (!tf_status.ok()) {
        SPDLOG_ERROR("Creating json from tensors failed: {}", tf_status.error_message());
        return StatusCode::REST_PROTO_TO_STRING_ERROR;
    }

    return StatusCode::OK;
}
}  // namespace ovms
