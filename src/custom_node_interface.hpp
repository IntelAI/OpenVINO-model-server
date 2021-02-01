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

#include <stdint.h>

enum CustomNodeTensorPrecision {
    UNSPECIFIED,
    FP32,
    FP16,
    U8,
    I8,
    I16,
    U16,
    I32
};

struct CustomNodeTensor {
    const char* name;
    uint8_t* data;
    uint64_t dataLength;
    uint64_t* dims;
    uint64_t dimsLength;
    CustomNodeTensorPrecision precision;
};

struct CustomNodeParam {
    const char *key, *value;
};

extern "C" int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength);
extern "C" int releaseBuffer(struct CustomNodeTensor* output);
extern "C" int releaseTensors(struct CustomNodeTensor* outputs);
