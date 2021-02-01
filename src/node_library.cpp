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
#include "node_library.hpp"

namespace ovms {

CustomNodeTensorPrecision toCustomNodeTensorPrecision(InferenceEngine::Precision precision) {
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        return CustomNodeTensorPrecision::FP32;
    case InferenceEngine::Precision::FP16:
        return CustomNodeTensorPrecision::FP16;
    case InferenceEngine::Precision::U8:
        return CustomNodeTensorPrecision::U8;
    case InferenceEngine::Precision::I8:
        return CustomNodeTensorPrecision::I8;
    case InferenceEngine::Precision::I16:
        return CustomNodeTensorPrecision::I16;
    case InferenceEngine::Precision::U16:
        return CustomNodeTensorPrecision::U16;
    case InferenceEngine::Precision::I32:
        return CustomNodeTensorPrecision::I32;
    default:
        return CustomNodeTensorPrecision::UNSPECIFIED;
    }
}

InferenceEngine::Precision toInferenceEnginePrecision(CustomNodeTensorPrecision precision) {
    switch (precision) {
    case CustomNodeTensorPrecision::FP32:
        return InferenceEngine::Precision::FP32;
    case CustomNodeTensorPrecision::FP16:
        return InferenceEngine::Precision::FP16;
    case CustomNodeTensorPrecision::U8:
        return InferenceEngine::Precision::U8;
    case CustomNodeTensorPrecision::I8:
        return InferenceEngine::Precision::I8;
    case CustomNodeTensorPrecision::I16:
        return InferenceEngine::Precision::I16;
    case CustomNodeTensorPrecision::U16:
        return InferenceEngine::Precision::U16;
    case CustomNodeTensorPrecision::I32:
        return InferenceEngine::Precision::I32;
    default:
        return InferenceEngine::Precision::UNSPECIFIED;
    }
}

}  // namespace ovms
