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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../custom_node_output_allocator.hpp"

using namespace ovms;

class NodeLibraryCheckingReleaseCalled {
public:
    static bool releaseBufferCalled;
    static int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength);
    static int releaseBuffer(struct CustomNodeTensor* output);
    static int releaseTensors(struct CustomNodeTensor* outputs);
};

int NodeLibraryCheckingReleaseCalled::execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    return 1;
}

int NodeLibraryCheckingReleaseCalled::releaseBuffer(struct CustomNodeTensor* output) {
    releaseBufferCalled = true;
    return 2;
}

int NodeLibraryCheckingReleaseCalled::releaseTensors(struct CustomNodeTensor* outputs) {
    return 3;
}

bool NodeLibraryCheckingReleaseCalled::releaseBufferCalled = false;

class CustomNodeOutputAllocatorCheckingFreeCalled : public CustomNodeOutputAllocator {
    bool freeCalled = false;

public:
    CustomNodeOutputAllocatorCheckingFreeCalled(struct CustomNodeTensor tensor, NodeLibrary nodeLibrary) :
        CustomNodeOutputAllocator(tensor, nodeLibrary) {
    }

    ~CustomNodeOutputAllocatorCheckingFreeCalled() {
        EXPECT_TRUE(freeCalled);
    }

    bool free(void* handle) noexcept override {
        bool tmp = CustomNodeOutputAllocator::free(handle);
        freeCalled = true;
        return tmp;
    }
};

TEST(CustomNodeOutputAllocator, BlobDeallocationCallsReleaseBuffer) {
    unsigned int elementsCount = 10;
    const InferenceEngine::TensorDesc desc{
        InferenceEngine::Precision::FP32,
        {elementsCount},
        InferenceEngine::Layout::C};
    std::vector<float> data(elementsCount);
    CustomNodeTensor tensor{
        "name",
        reinterpret_cast<uint8_t*>(data.data()),
        sizeof(float) * elementsCount,
        reinterpret_cast<uint64_t*>(&elementsCount),
        1,
        CustomNodeTensorPrecision::FP32};
    NodeLibrary library{
        NodeLibraryCheckingReleaseCalled::execute,
        NodeLibraryCheckingReleaseCalled::releaseBuffer,
        NodeLibraryCheckingReleaseCalled::releaseTensors};
    std::shared_ptr<CustomNodeOutputAllocator> customNodeOutputAllocator = std::make_shared<CustomNodeOutputAllocatorCheckingFreeCalled>(tensor, library);
    EXPECT_FALSE(NodeLibraryCheckingReleaseCalled::releaseBufferCalled);
    {
        InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(desc, customNodeOutputAllocator);
        blob->allocate();
    }
    EXPECT_TRUE(NodeLibraryCheckingReleaseCalled::releaseBufferCalled);
}

int execute(const struct CustomNodeTensor* inputs, int inputsLength, struct CustomNodeTensor** outputs, int* outputsLength, const struct CustomNodeParam* params, int paramsLength) {
    return 1;
}

int releaseBuffer(struct CustomNodeTensor* output) {
    return 2;
}

int releaseTensors(struct CustomNodeTensor* outputs) {
    return 3;
}

TEST(CustomNodeOutputAllocator, BlobReturnsCorrectPointer) {
    unsigned int elementsCount = 10;
    const InferenceEngine::TensorDesc desc{
        InferenceEngine::Precision::FP32,
        {elementsCount},
        InferenceEngine::Layout::C};
    std::vector<float> data(elementsCount);
    CustomNodeTensor tensor{
        "name",
        reinterpret_cast<uint8_t*>(data.data()),
        sizeof(float) * elementsCount,
        reinterpret_cast<uint64_t*>(&elementsCount),
        1,
        CustomNodeTensorPrecision::FP32};
    NodeLibrary library{
        execute,
        releaseBuffer,
        releaseTensors};
    std::shared_ptr<CustomNodeOutputAllocator> customNodeOutputAllocator = std::make_shared<CustomNodeOutputAllocator>(tensor, library);
    InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(desc, customNodeOutputAllocator);
    blob->allocate();
    EXPECT_EQ(static_cast<unsigned char*>(blob->buffer()), tensor.data);
}
