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
#pragma once

#include <map>
#include <string>

#include <rapidjson/document.h>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

/**
 * @brief Request order types
 */
enum class Order {
    UNKNOWN,
    ROW,
    COLUMN
};

/**
 * @brief Request format types
 */
enum class Format {
    UNKNOWN,
    NAMED,
    NONAMED
};

/**
 * @brief This class encapsulates http request body string parsing to request proto.
 */
class RestParser {
    /**
     * @brief Request order
     */
    Order order = Order::UNKNOWN;

    /**
     * @brief Request format
     */
    Format format = Format::UNKNOWN;

    /**
     * @brief Request proto
     */
    tensorflow::serving::PredictRequest requestProto;

    /**
     * @brief Request content precision
     */
    std::map<std::string, InferenceEngine::Precision> tensorPrecisionMap;

    void removeUnusedInputs();

    /**
     * @brief Increases batch size (0th-dimension) of tensor
     */
    static void increaseBatchSize(tensorflow::TensorProto& proto);

    /**
     * @brief Sets specific dimension to given size
     * 
     * @return returns false if dimension already existed and did not match requested size, true otherwise
     */
    static bool setDimOrValidate(tensorflow::TensorProto& proto, int dim, int size);

    /**
     * Parses and adds rapidjson value to tensor proto depending on underlying tensor data type
     */
    static bool addValue(tensorflow::TensorProto& proto, const rapidjson::Value& value);

    bool parseSequenceIdInput(rapidjson::Value& doc, tensorflow::TensorProto& proto, const std::string& tensorName);
    bool parseSequenceControlInput(rapidjson::Value& doc, tensorflow::TensorProto& proto, const std::string& tensorName);
    bool parseSpecialInput(rapidjson::Value& doc, tensorflow::TensorProto& proto, const std::string& tensorName);

    /**
     * @brief Parses rapidjson Node for arrays or numeric values on certain level of nesting.
     * 
     * @param doc rapidjson Node
     * @param dim level of nesting
     * @param input destination for parsing numeric values
     * 
     * @return false if processing failed, true when succeeded
     * 
     * Rapid json node expected to be passed in following structure:
     * [
     *     [...],
     *     [...],
     *     ...
     * ]
     */
    bool parseArray(rapidjson::Value& doc, int dim, tensorflow::TensorProto& proto, const std::string& tensorName);

    /**
     * @brief Parses rapidjson Node for inputs in a string(name)=>array(data) format
     * 
     * @param doc rapidjson Node
     * 
     * @return false if processing failed, true when succeeded
     * 
     * Rapid json node expected to be passed in following structure:
     * {
     *     "input1": [[...], [...], ...],
     *     "input2": [[...], [...], ...],
     *     ...
     * }
     */
    bool parseInstance(rapidjson::Value& doc);

    /**
     * @brief Checks whether all inputs have equal batch size, 0th-dimension
     * 
     * @return true or false
     */
    bool isBatchSizeEqualForAllInputs() const;

    /**
     * @brief Parses row format: list of objects, each object corresponding to one batch with one or multiple inputs.
     *        When no named format is detected, instance is treated as array of single input batches with no name.
     * 
     * @param node rapidjson Node
     * 
     * @return Status indicating if processing succeeded, error code otherwise
     *
     * Rapid json node expected to be passed in following structure:
     * [{inputs...}, {inputs...}, {inputs...}, ...]
     * or:
     * [no named input data batches...]
     */
    Status parseRowFormat(rapidjson::Value& node);

    /**
     * @brief Parses column format: object of input:batches key value pairs.
     *        When no named format is detected, instance is treated as array of single input batches with no name.
     * 
     * @param node rapidjson Node
     * 
     * @return Status indicating if processing succeeded, error code otherwise
     * 
     * Rapid json node expected to be passed in following structure:
     * {"inputA": [...], "inputB": [...], ...}
     * or:
     * [no named input data batches...]
     */
    Status parseColumnFormat(rapidjson::Value& node);

    bool setPrecisionIfNotSet(const rapidjson::Value& value, tensorflow::TensorProto& proto, const std::string& tensorName);

public:
    RestParser() = default;
    /**
     * @brief Constructor for preallocating memory for inputs beforehand. Size is calculated from tensor shape required by backend.
     * 
     * @param tensors Tensor map with model input parameters
     */
    RestParser(const tensor_map_t& tensors);

    /**
     * @brief Gets parsed request proto
     * 
     * @return proto
     */
    tensorflow::serving::PredictRequest& getProto() { return requestProto; }

    /**
     * @brief Gets request order
     */
    Order getOrder() const {
        return order;
    }

    /**
     * @brief Gets request format
     */
    Format getFormat() const {
        return format;
    }

    /**
     * @brief Parses http request body string
     * 
     * @param json request string
     * 
     * @return Status indicating error code or success
     * 
     * JSON expected to be passed in following structure:
     * {
     *     "signature_name": "serving_default",
     *     "instances": [
     *         {...}, {...}, {...}, ...
     *     ]
     * }
     */
    Status parse(const char* json);
};

}  // namespace ovms
