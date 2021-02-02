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
#pragma once

#include <memory>
#include <string>

#include "modelconfig.hpp"
#include "modelinstance.hpp"
#include "sequence_manager.hpp"

namespace ovms {

class StatefulModelInstance : public ModelInstance {
    static constexpr std::array<const char*, 2> SPECIAL_INPUT_NAMES{"sequence_id", "sequence_control_input"};
    std::unique_ptr<SequenceManager> sequenceManager;
    bool performLowLatencyTransformation;

public:
    /**
         * @brief A default constructor
         */
    StatefulModelInstance(const std::string& name, model_version_t version) :
        ModelInstance(name, version) {
        sequenceManager = std::make_unique<SequenceManager>(config.getSequenceTimeout(), config.getMaxSequenceNumber());
    }

    const std::unique_ptr<SequenceManager>& getSequenceManager() const {
        return this->sequenceManager;
    }

    const Status extractSequenceId(const tensorflow::TensorProto& proto, uint64_t& sequenceId);

    const Status extractSequenceControlInput(const tensorflow::TensorProto& proto, uint32_t& sequenceControlInput);
    /*
    Performs pre inference operations:
        - for SEQUENCE_START control input - reset InferRequest memory state
        - for SEQUENCE_END control input or for no control input - load sequence memory state into InferRequest

        Always returns StatusCode::OK
    */
    const Status preInferenceProcessing(InferenceEngine::InferRequest& inferRequest, SequenceProcessingSpec& sequenceProcessingSpec);

    /*
    Performs pre inference operations:
        - for SEQUENCE_START or for no control input - save InferRequest memory state in sequence memory state
        - for SEQUENCE_END control input - reset InferRequest memory state
        - for all requests - append sequence id to the response

        Always returns StatusCode::OK
    */
    const Status postInferenceProcessing(tensorflow::serving::PredictResponse* response,
        InferenceEngine::InferRequest& inferRequest, SequenceProcessingSpec& sequenceProcessingSpec);

    Status infer(const tensorflow::serving::PredictRequest* requestProto,
        tensorflow::serving::PredictResponse* responseProto,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) override;

protected:
    const Status validate(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& processingSpec);

    const Status validateNumberOfInputs(const tensorflow::serving::PredictRequest* request,
        const size_t expectedNumberOfInputs) override;

    Status loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter()) override;

private:
    const Status validateSpecialKeys(const tensorflow::serving::PredictRequest* request, SequenceProcessingSpec& sequenceProcessingSpec);
};
}  // namespace ovms
