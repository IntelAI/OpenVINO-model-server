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
#include <functional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../custom_node.hpp"
#include "../custom_node_library_manager.hpp"
#include "../dl_node.hpp"
#include "../entry_node.hpp"
#include "../exit_node.hpp"
#include "../node_library.hpp"
#include "test_utils.hpp"

using namespace ovms;
using namespace tensorflow;
using namespace tensorflow::serving;

class EnsembleFlowCustomNodePipelineExecutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        CustomNodeLibraryManager manager;
        ASSERT_EQ(manager.loadLibrary(
                      this->libraryName,
                      this->libraryPath),
            StatusCode::OK);
        ASSERT_EQ(manager.getLibrary(
                      this->libraryName,
                      this->library),
            StatusCode::OK);
    }

    template <typename T>
    void prepareRequest(const std::vector<T>& data) {
        tensorflow::TensorProto& proto = (*request.mutable_inputs())[pipelineInputName];
        proto.set_dtype(tensorflow::DataTypeToEnum<T>::value);
        proto.mutable_tensor_content()->assign((char*)data.data(), data.size() * sizeof(T));
        proto.mutable_tensor_shape()->add_dim()->set_size(1);
        proto.mutable_tensor_shape()->add_dim()->set_size(data.size());
    }

    template <typename T>
    std::unique_ptr<Pipeline> prepareSingleNodePipelineWithLibraryMock() {
        const std::vector<float> inputValues{3.5, 2.1, -0.2};
        this->prepareRequest(inputValues);
        auto input_node = std::make_unique<EntryNode>(&request);
        auto output_node = std::make_unique<ExitNode>(&response);
        auto custom_node = std::make_unique<CustomNode>(
            customNodeName,
            createLibraryMock<T>(),
            parameters_t{});

        auto pipeline = std::make_unique<Pipeline>(*input_node, *output_node);
        pipeline->connect(*input_node, *custom_node, {{pipelineInputName, customNodeInputName}});
        pipeline->connect(*custom_node, *output_node, {{customNodeOutputName, pipelineOutputName}});

        pipeline->push(std::move(input_node));
        pipeline->push(std::move(custom_node));
        pipeline->push(std::move(output_node));
        return pipeline;
    }

    template <typename T>
    void checkResponse(std::vector<T> data, std::function<T(T)> op) {
        this->checkResponse(this->pipelineOutputName, data, op);
    }

    template <typename T>
    void checkResponse(const std::string& outputName, std::vector<T> data, std::function<T(T)> op) {
        std::transform(data.begin(), data.end(), data.begin(), op);
        ASSERT_TRUE(response.outputs().contains(outputName));
        const auto& proto = response.outputs().at(outputName);

        ASSERT_EQ(proto.tensor_content().size(), data.size() * sizeof(T));
        ASSERT_EQ(proto.tensor_shape().dim_size(), 2);
        ASSERT_EQ(proto.tensor_shape().dim(0).size(), 1);
        ASSERT_EQ(proto.tensor_shape().dim(1).size(), data.size());

        auto* ptr = reinterpret_cast<const T*>(proto.tensor_content().c_str());

        const std::vector<T> actual(ptr, ptr + data.size());

        for (size_t i = 0; i < actual.size(); i++) {
            EXPECT_NEAR(actual[i], data[i], 0.001);
        }
    }

    template <typename T>
    static NodeLibrary createLibraryMock() {
        return NodeLibrary{
            T::execute,
            T::releaseBuffer,
            T::releaseTensors};
    }

    PredictRequest request;
    PredictResponse response;

    NodeLibrary library;

    const std::string customNodeName = "add_sub_node";
    const std::string libraryName = "add_sub_lib";
    const std::string libraryPath = "/ovms/bazel-bin/src/lib_node_add_sub.so";
    const std::string customNodeInputName = "input_numbers";
    const std::string customNodeOutputName = "output_numbers";
    const std::string pipelineInputName = "pipeline_input";
    const std::string pipelineOutputName = "pipeline_output";
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, AddSubCustomNode) {
    // Most basic configuration, just process single add-sub custom node pipeline request
    // input  add-sub  output
    //  O------->O------->O
    const std::vector<float> inputValues{3.2, 5.7, -2.4};
    this->prepareRequest(inputValues);

    const float addValue = 2.5;
    const float subValue = 4.8;

    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);
    auto custom_node = std::make_unique<CustomNode>(customNodeName, library,
        parameters_t{
            {"add_value", std::to_string(addValue)},
            {"sub_value", std::to_string(subValue)}});

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *custom_node, {{pipelineInputName, customNodeInputName}});
    pipeline.connect(*custom_node, *output_node, {{customNodeOutputName, pipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(custom_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);

    this->checkResponse<float>(inputValues, [addValue, subValue](float value) -> float {
        return value + addValue - subValue;
    });
}

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, SeriesOfCustomNodes) {
    constexpr int N = 100;
    constexpr int PARAMETERS = 2;
    static_assert(PARAMETERS > 0);
    static_assert(N > PARAMETERS);
    static_assert((N % PARAMETERS) == 0);
    // input      add-sub x N      output
    //  O------->O->O...O->O------->O

    const std::vector<float> inputValues{3.2, 5.7, -2.4};
    this->prepareRequest(inputValues);

    const std::array<float, PARAMETERS> addValue{1.5, -2.4};
    const std::array<float, PARAMETERS> subValue{-5.1, 1.9};

    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);

    std::unique_ptr<CustomNode> custom_nodes[N];
    for (int i = 0; i < N; i++) {
        custom_nodes[i] = std::make_unique<CustomNode>(customNodeName + std::to_string(i), library,
            parameters_t{
                {"add_value", std::to_string(addValue[i % PARAMETERS])},
                {"sub_value", std::to_string(subValue[i % PARAMETERS])}});
    }

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *(custom_nodes[0]), {{pipelineInputName, customNodeInputName}});
    pipeline.connect(*(custom_nodes[N - 1]), *output_node, {{customNodeOutputName, pipelineOutputName}});
    for (int i = 0; i < N - 1; i++) {
        pipeline.connect(*(custom_nodes[i]), *(custom_nodes[i + 1]), {{customNodeOutputName, customNodeInputName}});
    }

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));
    for (auto& custom_node : custom_nodes) {
        pipeline.push(std::move(custom_node));
    }

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);

    this->checkResponse<float>(inputValues, [N, addValue, subValue](float value) -> float {
        for (int i = 0; i < PARAMETERS; i++) {
            value += (N / PARAMETERS) * addValue[i];
            value -= (N / PARAMETERS) * subValue[i];
        }
        return value;
    });
}

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, ParallelCustomNodes) {
    constexpr int N = 200;
    constexpr int PARAMETERS = 5;
    static_assert(PARAMETERS > 0);
    static_assert(N > PARAMETERS);
    static_assert((N % PARAMETERS) == 0);
    /* input    add-sub x N      output
        O---------->O------------->O
        ...        ...            /\
        L---------->O-------------_|
    */

    const std::vector<float> inputValues{9.1, -3.7, 22.2};
    this->prepareRequest(inputValues);

    const std::array<float, PARAMETERS> addValue{4.5, 0.2, -0.6, 0.4, -2.5};
    const std::array<float, PARAMETERS> subValue{8.5, -3.2, 10.0, -0.5, 2.4};

    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);

    Pipeline pipeline(*input_node, *output_node);
    std::unique_ptr<CustomNode> custom_nodes[N];
    for (int i = 0; i < N; i++) {
        custom_nodes[i] = std::make_unique<CustomNode>(customNodeName + std::to_string(i), library,
            parameters_t{
                {"add_value", std::to_string(addValue[i % PARAMETERS])},
                {"sub_value", std::to_string(subValue[i % PARAMETERS])}});
        pipeline.connect(*input_node, *(custom_nodes[i]),
            {{pipelineInputName, customNodeInputName}});
        pipeline.connect(*(custom_nodes[i]), *output_node,
            {{customNodeOutputName, pipelineOutputName + std::to_string(i)}});
        pipeline.push(std::move(custom_nodes[i]));
    }
    pipeline.push(std::move(input_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), N);

    for (int i = 0; i < N; i++) {
        this->checkResponse<float>(
            pipelineOutputName + std::to_string(i),
            inputValues,
            [i, addValue, subValue](float value) -> float {
                value += addValue[i % PARAMETERS];
                value -= subValue[i % PARAMETERS];
                return value;
            });
    }
}

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, CustomAndDLNodes) {
    // input  add-sub1 dummy  add-sub2 output
    //  O------->O------O--------O------>O
    ConstructorEnabledModelManager modelManager;
    ModelConfig config = DUMMY_MODEL_CONFIG;
    modelManager.reloadModelWithVersions(config);

    const std::vector<float> inputValues{
        4, 1.5, -5, -2.5, 9.3, 0.3, -0.15, 7.4, 5.2, -2.4};
    this->prepareRequest(inputValues);

    const float addValue[] = {-0.85, 30.2};
    const float subValue[] = {1.35, -28.5};

    auto input_node = std::make_unique<EntryNode>(&request);
    auto output_node = std::make_unique<ExitNode>(&response);
    auto model_node = std::make_unique<DLNode>(
        "dummy_node",
        "dummy",
        std::nullopt,
        modelManager);
    std::unique_ptr<CustomNode> custom_node[] = {
        std::make_unique<CustomNode>(customNodeName + "_0", library,
            parameters_t{
                {"add_value", std::to_string(addValue[0])},
                {"sub_value", std::to_string(subValue[0])}}),
        std::make_unique<CustomNode>(customNodeName + "_1", library,
            parameters_t{
                {"add_value", std::to_string(addValue[1])},
                {"sub_value", std::to_string(subValue[1])}})};

    Pipeline pipeline(*input_node, *output_node);
    pipeline.connect(*input_node, *(custom_node[0]), {{pipelineInputName, customNodeInputName}});
    pipeline.connect(*(custom_node[0]), *model_node, {{customNodeOutputName, DUMMY_MODEL_INPUT_NAME}});
    pipeline.connect(*model_node, *(custom_node[1]), {{DUMMY_MODEL_OUTPUT_NAME, customNodeInputName}});
    pipeline.connect(*(custom_node[1]), *output_node, {{customNodeOutputName, pipelineOutputName}});

    pipeline.push(std::move(input_node));
    pipeline.push(std::move(custom_node[0]));
    pipeline.push(std::move(custom_node[1]));
    pipeline.push(std::move(model_node));
    pipeline.push(std::move(output_node));

    ASSERT_EQ(pipeline.execute(), StatusCode::OK);
    ASSERT_EQ(response.outputs().size(), 1);

    this->checkResponse<float>(inputValues, [addValue, subValue](float value) -> float {
        return value + 1.0 + addValue[0] + addValue[1] - subValue[0] - subValue[1];
    });
}

struct LibraryFailInExecute {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor**, int*, const struct CustomNodeParam*, int) {
        return 1;
    }
    static int releaseBuffer(struct CustomNodeTensor*) {
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor*) {
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeExecution) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryFailInExecute>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_EXECUTION_FAILED);
}

struct LibraryCorruptedOutputHandle {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = nullptr;
        *outputsNum = 5;
        return 0;
    }
    static int releaseBuffer(struct CustomNodeTensor*) {
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor*) {
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputsCorruptedHandle) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryCorruptedOutputHandle>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED);
}

struct LibraryCorruptedOutputsNumber {
    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)0x004def;
        *outputsNum = 0;
        return 0;
    }
    static int releaseBuffer(struct CustomNodeTensor*) {
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor*) {
        return 0;
    }
};

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputsCorruptedNumberOfOutputs) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryCorruptedOutputsNumber>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT);
}

struct LibraryMissingOutput {
    static bool releaseBufferCalled;

    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "random_not_connected_output";
        (*handle)->precision = CustomNodeTensorPrecision::FP32;
        (*handle)->dims = (uint64_t*)malloc(sizeof(uint64_t));
        (*handle)->dims[0] = 1;
        (*handle)->dimsLength = 1;
        (*handle)->data = (uint8_t*)malloc(sizeof(float) * sizeof(uint8_t));
        (*handle)->dataLength = sizeof(float);
        return 0;
    }
    static int releaseBuffer(struct CustomNodeTensor* tensor) {
        releaseBufferCalled = true;
        free(tensor->dims);
        free(tensor->data);
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor* handle) {
        free(handle);
        return 0;
    }
};

bool LibraryMissingOutput::releaseBufferCalled = false;

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeMissingOutput) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryMissingOutput>();
    ASSERT_EQ(pipeline->execute(), StatusCode::NODE_LIBRARY_MISSING_OUTPUT);
    ASSERT_TRUE(LibraryMissingOutput::releaseBufferCalled);
}

struct LibraryIncorrectOutputPrecision {
    static bool releaseBufferCalled;

    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "output_numbers";
        (*handle)->precision = CustomNodeTensorPrecision::UNSPECIFIED;
        (*handle)->dims = (uint64_t*)malloc(sizeof(uint64_t));
        (*handle)->dimsLength = 1;
        (*handle)->data = (uint8_t*)malloc(sizeof(uint8_t));
        (*handle)->dataLength = 1;
        return 0;
    }
    static int releaseBuffer(struct CustomNodeTensor* tensor) {
        releaseBufferCalled = true;
        free(tensor->dims);
        free(tensor->data);
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor* handle) {
        free(handle);
        return 0;
    }
};

bool LibraryIncorrectOutputPrecision::releaseBufferCalled = false;

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputInvalidPrecision) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryIncorrectOutputPrecision>();
    ASSERT_EQ(pipeline->execute(), StatusCode::INVALID_PRECISION);
    ASSERT_TRUE(LibraryIncorrectOutputPrecision::releaseBufferCalled);
}

struct LibraryIncorrectOutputShape {
    static bool releaseBufferCalled;

    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "output_numbers";
        (*handle)->precision = CustomNodeTensorPrecision::FP32;
        (*handle)->dims = nullptr;
        (*handle)->dimsLength = 0;
        (*handle)->data = (uint8_t*)malloc(sizeof(uint8_t));
        (*handle)->dataLength = 1;
        return 0;
    }
    static int releaseBuffer(struct CustomNodeTensor* tensor) {
        free(tensor->data);
        releaseBufferCalled = true;
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor* handle) {
        free(handle);
        return 0;
    }
};

bool LibraryIncorrectOutputShape::releaseBufferCalled = false;

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputInvalidShape) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryIncorrectOutputShape>();
    ASSERT_EQ(pipeline->execute(), StatusCode::INVALID_SHAPE);
    ASSERT_TRUE(LibraryIncorrectOutputShape::releaseBufferCalled);
}

struct LibraryIncorrectOutputContentSize {
    static bool releaseBufferCalled;

    static int execute(const struct CustomNodeTensor*, int, struct CustomNodeTensor** handle, int* outputsNum, const struct CustomNodeParam*, int) {
        *handle = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor));
        *outputsNum = 1;
        (*handle)->name = "output_numbers";
        (*handle)->precision = CustomNodeTensorPrecision::FP32;
        (*handle)->dims = (uint64_t*)malloc(sizeof(uint64_t));
        (*handle)->dimsLength = 1;
        (*handle)->data = nullptr;
        (*handle)->dataLength = 0;
        return 0;
    }
    static int releaseBuffer(struct CustomNodeTensor* tensor) {
        free(tensor->dims);
        releaseBufferCalled = true;
        return 0;
    }
    static int releaseTensors(struct CustomNodeTensor* handle) {
        free(handle);
        return 0;
    }
};

bool LibraryIncorrectOutputContentSize::releaseBufferCalled = false;

TEST_F(EnsembleFlowCustomNodePipelineExecutionTest, FailInCustomNodeOutputInvalidContentSize) {
    auto pipeline = this->prepareSingleNodePipelineWithLibraryMock<LibraryIncorrectOutputContentSize>();
    ASSERT_EQ(pipeline->execute(), StatusCode::INVALID_CONTENT_SIZE);
    ASSERT_TRUE(LibraryIncorrectOutputContentSize::releaseBufferCalled);
}

class EnsembleFlowCustomNodeFactoryCreateThenExecuteTest : public ::testing::Test {};
TEST_F(EnsembleFlowCustomNodeFactoryCreateThenExecuteTest, SimplePipelineFactoryCreationWithCustomNode) {
    ASSERT_TRUE(false);
}

TEST_F(EnsembleFlowCustomNodeFactoryCreateThenExecuteTest, ParallelPipelineFactoryUsageWithCustomNode) {
    ASSERT_TRUE(false);
}

class EnsembleFlowCustomNodeLoadConfigThenExecuteTest : public TestWithTempDir {};

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, AddSubCustomNode) {
    ASSERT_TRUE(false);
}

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, ReferenceMissingLibrary) {
    ASSERT_TRUE(false);
}

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, ReferenceCorruptedLibrary) {
    ASSERT_TRUE(false);
}

TEST_F(EnsembleFlowCustomNodeLoadConfigThenExecuteTest, MissingRequiredNodeParameters) {
    ASSERT_TRUE(false);
}

// Library with not escaped path
// Reload config tests

// TODO: Validation tests (PipelineDefinition::validateNodes/validateForCycles)
