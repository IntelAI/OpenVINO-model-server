//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#include <utility>

#include <grpcpp/server_context.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/util/net_http/server/public/response_code_enum.h"
#pragma GCC diagnostic pop

namespace ovms {

namespace net_http = tensorflow::serving::net_http;

enum class StatusCode {
    OK, /*!< Success */

    PATH_INVALID,     /*!< The provided path is invalid or doesn't exists */
    FILE_INVALID,     /*!< File not found or cannot open */
    FILESYSTEM_ERROR, /*!< Underlaying filesystem error */
    NETWORK_NOT_LOADED,
    JSON_INVALID,             /*!< The file/content is not valid json */
    JSON_SERIALIZATION_ERROR, /*!< Data serialization to json format failed */
    MODELINSTANCE_NOT_FOUND,
    SHAPE_WRONG_FORMAT,                   /*!< The provided shape param is in wrong format */
    PLUGIN_CONFIG_WRONG_FORMAT,           /*!< Plugin config is in wrong format */
    MODEL_VERSION_POLICY_WRONG_FORMAT,    /*!< Model version policy is in wrong format */
    MODEL_VERSION_POLICY_UNSUPPORTED_KEY, /*!< Model version policy contains invalid key */
    GRPC_CHANNEL_ARG_WRONG_FORMAT,
    NO_MODEL_VERSION_AVAILABLE,             /*!< No model version found in path */
    RESHAPE_ERROR,                          /*!< Impossible to perform reshape */
    RESHAPE_REQUIRED,                       /*!< Model instance needs to be reloaded with new shape */
    BATCHSIZE_CHANGE_REQUIRED,              /*!< Model instance needs to be reloaded with new batch size */
    FORBIDDEN_MODEL_DYNAMIC_PARAMETER,      /*!< Value of the provided param is forbidden */
    ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED,      /*!< Anonymous fixed shape is invalid for models with multiple inputs */
    CONFIG_SHAPE_IS_NOT_IN_NETWORK,         /*!< Invalid shape dimension number or dimension value */
    CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE, /*!< Cannot load network into target device */
    REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL,

    // Model management
    MODEL_MISSING,                    /*!< Model with such name and/or version does not exist */
    MODEL_NAME_MISSING,               /*!< Model with requested name is not found */
    MODEL_VERSION_MISSING,            /*!< Model with requested version is not found */
    MODEL_VERSION_NOT_LOADED_ANYMORE, /*!< Model with requested version is retired */
    MODEL_VERSION_NOT_LOADED_YET,     /*!< Model with requested version is not loaded yet */
    INVALID_NIREQ,                    /*!< Invalid NIREQ requested */

    // Sequence management
    SEQUENCE_MISSING,                /*!< Sequence with provided ID does not exist */
    SEQUENCE_ALREADY_EXISTS,         /*!< Sequence with provided ID already exists */
    SEQUENCE_ID_NOT_PROVIDED,        /*!< Sequence ID has not been provided in request inputs */
    SEQUENCE_ID_BAD_TYPE,            /*!< Wrong sequence ID type */
    INVALID_SEQUENCE_CONTROL_INPUT,  /*!< Unexpected value of sequence control input */
    SEQUENCE_CONTROL_INPUT_BAD_TYPE, /*!< Sequence control input in bad type*/
    SEQUENCE_TERMINATED,             /*!< Sequence last request is being processed and it's not available anymore*/

    // Predict request validation
    INVALID_NO_OF_INPUTS,           /*!< Invalid number of inputs */
    INVALID_MISSING_INPUT,          /*!< Missing one or more of inputs */
    INVALID_MISSING_OUTPUT,         /*!< Missing one or more of outputs */
    INVALID_NO_OF_SHAPE_DIMENSIONS, /*!< Invalid number of shape dimensions */
    INVALID_BATCH_SIZE,             /*!< Input batch size other than required */
    INVALID_SHAPE,                  /*!< Invalid shape dimension number or dimension value */
    INVALID_PRECISION,              /*!< Invalid precision */
    INVALID_VALUE_COUNT,            /*!< Invalid value count error status for uint16 and half float data types */
    INVALID_CONTENT_SIZE,           /*!< Invalid content size error status for types using tensor_content() */

    // Deserialization
    OV_UNSUPPORTED_DESERIALIZATION_PRECISION, /*!< Unsupported deserialization precision, theoretically should never be returned since ModelInstance::validation checks against network precision */
    OV_INTERNAL_DESERIALIZATION_ERROR,        /*!< Error occured during deserialization */

    // Inference
    OV_INTERNAL_INFERENCE_ERROR, /*!< Error occured during inference */

    // Serialization
    OV_UNSUPPORTED_SERIALIZATION_PRECISION, /*!< Unsupported serializaton precision */
    OV_INTERNAL_SERIALIZATION_ERROR,        /*!< Error occurred during serialization */
    OV_CLONE_BLOB_ERROR,                    /*!< Error during blob clone */

    // GetModelStatus
    INVALID_SIGNATURE_DEF, /*!< Requested signature is not supported */

    // Common request validation errors
    MODEL_SPEC_MISSING, /*!< Request lacks model_spec */

    INTERNAL_ERROR,

    UNKNOWN_ERROR,

    NOT_IMPLEMENTED,

    // S3
    S3_BUCKET_NOT_FOUND, /*!< S3 Bucket not found  */
    S3_METADATA_FAIL,
    S3_FAILED_LIST_OBJECTS,
    S3_FAILED_GET_TIME,
    S3_INVALID_ACCESS,
    S3_FILE_NOT_FOUND,
    S3_FILE_INVALID,
    S3_FAILED_GET_OBJECT,

    // GCS
    GCS_BUCKET_NOT_FOUND,
    GCS_METADATA_FAIL,
    GCS_FAILED_LIST_OBJECTS,
    GCS_FAILED_GET_TIME,
    GCS_INVALID_ACCESS,
    GCS_FILE_NOT_FOUND,
    GCS_FILE_INVALID,
    GCS_FAILED_GET_OBJECT,
    GCS_INCORRECT_REQUESTED_OBJECT_TYPE,

    // AS
    AS_INVALID_PATH,
    AS_CONTAINER_NOT_FOUND,
    AS_SHARE_NOT_FOUND,
    AS_METADATA_FAIL,
    AS_FAILED_LIST_OBJECTS,
    AS_FAILED_GET_TIME,
    AS_INVALID_ACCESS,
    AS_FILE_NOT_FOUND,
    AS_FILE_INVALID,
    AS_FAILED_GET_OBJECT,
    AS_INCORRECT_REQUESTED_OBJECT_TYPE,

    // REST handler
    REST_NOT_FOUND,               /*!< Requested REST resource not found */
    REST_COULD_NOT_PARSE_VERSION, /*!< Could not parse model version in request */
    REST_INVALID_URL,             /*!< Malformed REST request url */
    REST_UNSUPPORTED_METHOD,      /*!< Request sent with unsupported method */
    REST_MALFORMED_REQUEST,       /*!< Malformed REST request */

    // REST Parse
    REST_BODY_IS_NOT_AN_OBJECT,          /*!< REST body should be JSON object */
    REST_PREDICT_UNKNOWN_ORDER,          /*!< Could not detect order (row/column) */
    REST_INSTANCES_NOT_AN_ARRAY,         /*!< When parsing row order, instances must be an array */
    REST_NAMED_INSTANCE_NOT_AN_OBJECT,   /*!< When parsing named instance it needs to be an object */
    REST_INPUT_NOT_PREALLOCATED,         /*!< When parsing no named instance, exactly one input need to be preallocated */
    REST_NO_INSTANCES_FOUND,             /*!< Missing instances in row order */
    REST_INSTANCES_NOT_NAMED_OR_NONAMED, /*!< Unknown instance format, neither named or nonamed */
    REST_COULD_NOT_PARSE_INSTANCE,       /*!< Error while parsing instance content, not valid ndarray */
    REST_INSTANCES_BATCH_SIZE_DIFFER,    /*!< In row order 0-th dimension (batch size) must be equal for all inputs */
    REST_INPUTS_NOT_AN_OBJECT,           /*!< When parsing column order, inputs must be an object */
    REST_NO_INPUTS_FOUND,                /*!< Missing inputs in column order */
    REST_COULD_NOT_PARSE_INPUT,          /*!< Error while parsing input content, not valid ndarray */
    REST_PROTO_TO_STRING_ERROR,          /*!< Error while parsing ResponseProto to JSON string */
    REST_UNSUPPORTED_PRECISION,          /*!< Unsupported conversion from tensor_content to _val container */
    REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE,

    // Pipeline validation errors
    PIPELINE_DEFINITION_ALREADY_EXIST,
    PIPELINE_NODE_WRONG_KIND_CONFIGURATION,
    PIPELINE_MULTIPLE_ENTRY_NODES,
    PIPELINE_MULTIPLE_EXIT_NODES,
    PIPELINE_MISSING_ENTRY_OR_EXIT,
    PIPELINE_DEFINITION_NAME_MISSING,
    PIPELINE_DEFINITION_NOT_LOADED_ANYMORE,
    PIPELINE_DEFINITION_NOT_LOADED_YET,
    PIPELINE_NODE_NAME_DUPLICATE,
    PIPELINE_STREAM_ID_NOT_READY_YET,
    PIPELINE_CYCLE_FOUND,
    PIPELINE_CONTAINS_UNCONNECTED_NODES,
    PIPELINE_NODE_REFERING_TO_MISSING_NODE,
    PIPELINE_NODE_REFERING_TO_MISSING_MODEL,
    PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE,
    PIPELINE_NODE_REFERING_TO_MISSING_MODEL_OUTPUT,
    PIPELINE_CONNECTION_TO_MISSING_MODEL_INPUT,
    PIPELINE_NOT_ALL_INPUTS_CONNECTED,
    PIPELINE_MODEL_INPUT_CONNECTED_TO_MULTIPLE_DATA_SOURCES,
    PIPELINE_EXIT_USED_AS_NODE_DEPENDENCY,
    PIPELINE_NAME_OCCUPIED,

    // Custom Loader
    CUSTOM_LOADER_LIBRARY_INVALID,
    CUSTOM_LOADER_LIBRARY_LOAD_FAILED,
    CUSTOM_LOADER_EXISTS,
    CUSTOM_LOADER_NOT_PRESENT,
    CUSTOM_LOADER_INIT_FAILED,
    CUSTOM_LOADER_ERROR,

    // Custom Node
    NODE_LIBRARY_ALREADY_LOADED,
    NODE_LIBRARY_LOAD_FAILED_OPEN,
    NODE_LIBRARY_LOAD_FAILED_SYM,
    NODE_LIBRARY_MISSING,
    NODE_LIBRARY_MISSING_OUTPUT,
    NODE_LIBRARY_EXECUTION_FAILED,
    NODE_LIBRARY_OUTPUTS_CORRUPTED,
    NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT,

    STATUS_CODE_END
};

class Status {
    StatusCode code;
    std::string message;

    static const std::map<const StatusCode, const std::string> statusMessageMap;
    static const std::map<const StatusCode, grpc::StatusCode> grpcStatusMap;
    static const std::map<const StatusCode, net_http::HTTPStatusCode> httpStatusMap;

    void appendDetails(const std::string& details) {
        this->message += " - " + details;
    }

public:
    Status(StatusCode code = StatusCode::OK) :
        code(code) {
        auto it = statusMessageMap.find(code);
        if (it != statusMessageMap.end())
            this->message = it->second;
        else
            this->message = "Unknown error";
    }

    Status(StatusCode code, const std::string& details) :
        Status(code) {
        appendDetails(details);
    }

    bool ok() const {
        return code == StatusCode::OK;
    }

    const StatusCode getCode() const {
        return this->code;
    }

    bool batchSizeChangeRequired() const {
        return code == StatusCode::BATCHSIZE_CHANGE_REQUIRED;
    }

    bool reshapeRequired() const {
        return code == StatusCode::RESHAPE_REQUIRED;
    }

    bool operator==(const Status& status) const {
        return this->code == status.code;
    }

    bool operator!=(const Status& status) const {
        return this->code != status.code;
    }

    const grpc::Status grpc() const {
        auto it = grpcStatusMap.find(code);
        if (it != grpcStatusMap.end()) {
            return grpc::Status(it->second, this->message);
        } else {
            return grpc::Status(grpc::StatusCode::UNKNOWN, "Unknown error");
        }
    }

    operator grpc::Status() const {
        return this->grpc();
    }

    const std::string& string() const {
        return this->message;
    }

    operator const std::string&() const {
        return this->string();
    }

    const net_http::HTTPStatusCode http() const {
        auto it = httpStatusMap.find(code);
        if (it != httpStatusMap.end()) {
            return it->second;
        } else {
            return net_http::HTTPStatusCode::ERROR;
        }
    }
};

}  // namespace ovms
