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

#include "status.hpp"

namespace ovms {

const std::map<const StatusCode, const std::string> Status::statusMessageMap = {
    {StatusCode::OK, ""},

    {StatusCode::PATH_INVALID, "The provided base path is invalid or doesn't exists"},
    {StatusCode::FILE_INVALID, "File not found or cannot open"},
    {StatusCode::NO_MODEL_VERSION_AVAILABLE, "Not a single model version directory has valid numeric name"},
    {StatusCode::NETWORK_NOT_LOADED, "Error while loading a network"},
    {StatusCode::JSON_INVALID, "The file is not valid json"},
    {StatusCode::MODELINSTANCE_NOT_FOUND, "ModelInstance not found"},
    {StatusCode::SHAPE_WRONG_FORMAT, "The provided shape is in wrong format"},
    {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, "Plugin config is in wrong format"},
    {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, "Model version policy is in wrong format"},
    {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, "Model version policy contains unsupported key"},
    {StatusCode::RESHAPE_ERROR, "Model could not be reshaped with requested shape"},
    {StatusCode::ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED, "Anonymous fixed shape is invalid for models with multiple inputs"},
    {StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE, "Cannot load network into target device"},
    {StatusCode::MODEL_MISSING, "Model with requested name and/or version is not found"},
    {StatusCode::MODEL_NAME_MISSING, "Model with requested name is not found"},
    {StatusCode::MODEL_VERSION_MISSING, "Model with requested version is not found"},
    {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, "Model with requested version is retired"},
    {StatusCode::MODEL_VERSION_NOT_LOADED_YET, "Model with requested version is not loaded yet"},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, "Pipeline is retired"},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, "Pipeline is not loaded yet"},
    {StatusCode::MODEL_SPEC_MISSING, "model_spec missing in request"},
    {StatusCode::INVALID_SIGNATURE_DEF, "Invalid signature name"},
    {StatusCode::CONFIG_SHAPE_IS_NOT_IN_NETWORK, "Shape from config not found in network"},
    {StatusCode::INVALID_NIREQ, "Nireq parameter too high"},
    {StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL, "Requested dynamic parameters but model is subscribed to pipeline"},
    {StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET, "Node is not ready for execution"},

    // Sequence management
    {StatusCode::SEQUENCE_MISSING, "Sequence with provided ID does not exist"},
    {StatusCode::SEQUENCE_ALREADY_EXISTS, "Sequence with provided ID already exists"},
    {StatusCode::SEQUENCE_ID_NOT_PROVIDED, "Sequence ID has not been provided in request inputs"},
    {StatusCode::INVALID_SEQUENCE_CONTROL_INPUT, "Unexpected value of sequence control input"},
    {StatusCode::SEQUENCE_ID_BAD_TYPE, "Could not find sequence id in expected tensor proto field uint64_val"},
    {StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE, "Could not find sequence control input in expected tensor proto field uint32_val"},
    {StatusCode::SEQUENCE_TERMINATED, "Sequence last request is being processed and it's not available anymore"},
    {StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE, "Special input proto does not contain tensor shape information"},

    // Predict request validation
    {StatusCode::INVALID_NO_OF_INPUTS, "Invalid number of inputs"},
    {StatusCode::INVALID_MISSING_INPUT, "Missing input with specific name"},
    {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, "Invalid number of shape dimensions"},
    {StatusCode::INVALID_BATCH_SIZE, "Invalid input batch size"},
    {StatusCode::INVALID_SHAPE, "Invalid input shape"},
    {StatusCode::INVALID_PRECISION, "Invalid input precision"},
    {StatusCode::INVALID_VALUE_COUNT, "Invalid number of values in tensor proto container"},
    {StatusCode::INVALID_CONTENT_SIZE, "Invalid content size of tensor proto"},

    // Deserialization
    {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, "Unsupported deserialization precision"},
    {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, "Internal deserialization error"},

    // Inference
    {StatusCode::OV_INTERNAL_INFERENCE_ERROR, "Internal inference error"},

    // Serialization
    {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, "Unsupported serialization precision"},
    {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, "Internal serialization error"},

    // GetModelStatus
    {StatusCode::INTERNAL_ERROR, "Internal server error"},

    // Rest handler failure
    {StatusCode::REST_INVALID_URL, "Invalid request URL"},
    {StatusCode::REST_UNSUPPORTED_METHOD, "Unsupported method"},

    // Rest parser failure
    {StatusCode::REST_BODY_IS_NOT_AN_OBJECT, "Request body should be JSON object"},
    {StatusCode::REST_PREDICT_UNKNOWN_ORDER, "Invalid JSON structure. Could not detect row or column format"},
    {StatusCode::REST_INSTANCES_NOT_AN_ARRAY, "Invalid JSON structure. Nonamed instance is not an array."},
    {StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT, "Invalid JSON structure. One of named instances is not a JSON object."},
    {StatusCode::REST_INPUT_NOT_PREALLOCATED, "Internal allocation error"},
    {StatusCode::REST_NO_INSTANCES_FOUND, "Invalid JSON structure. Missing instances in row format"},
    {StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED, "Could not detect neither named or nonamed format"},
    {StatusCode::REST_COULD_NOT_PARSE_INSTANCE, "Could not parse instance content. Not valid ndarray detected"},
    {StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER, "Invalid JSON structure. Request inputs have different batch sizes"},
    {StatusCode::REST_INPUTS_NOT_AN_OBJECT, "Invalid JSON structure. One of inputs is not a JSON object."},
    {StatusCode::REST_NO_INPUTS_FOUND, "Invalid JSON structure. Missing inputs in column format"},
    {StatusCode::REST_COULD_NOT_PARSE_INPUT, "Could not parse input content. Not valid ndarray detected"},
    {StatusCode::REST_PROTO_TO_STRING_ERROR, "Response parsing to JSON error"},
    {StatusCode::REST_UNSUPPORTED_PRECISION, "Could not parse input content. Unsupported data precision detected"},
    {StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE, "Size of data in tensor_content does not match declared tensor shape"},
    {StatusCode::REST_SERIALIZE_VAL_FIELD_INVALID_SIZE, "Number of elements in xxx_val field does not match declared tensor shape"},
    {StatusCode::REST_SERIALIZE_NO_DATA, "No data found in tensor_content or xxx_val field matching tensor dtype"},

    // Pipeline validation errors
    {StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST, "Pipeline definition with the same name already exists"},
    {StatusCode::PIPELINE_NODE_WRONG_KIND_CONFIGURATION, "Unsupported node type"},
    {StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES, "Pipeline definition has multiple request nodes"},
    {StatusCode::PIPELINE_MULTIPLE_EXIT_NODES, "Pipeline definition has multiple response nodes"},
    {StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT, "Pipeline definition is missing request or response node"},
    {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, "Model with requested name is not found"},
    {StatusCode::PIPELINE_NODE_NAME_DUPLICATE, "Pipeline definition has multiple nodes with the same name"},
    {StatusCode::PIPELINE_CYCLE_FOUND, "Pipeline definition contains a cycle"},
    {StatusCode::PIPELINE_CONTAINS_UNCONNECTED_NODES, "Pipeline definition has unconnected nodes"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_NODE, "Pipeline definition has reference to missing node"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL, "Pipeline definition has reference to missing model"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE, "Pipeline definition has reference to missing data source"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL_OUTPUT, "Pipeline definition has reference to missing model output"},
    {StatusCode::PIPELINE_CONNECTION_TO_MISSING_MODEL_INPUT, "Pipeline definition has connection to non existing model input"},
    {StatusCode::PIPELINE_NOT_ALL_INPUTS_CONNECTED, "Pipeline definition does not have connections for all inputs of underlying models"},
    {StatusCode::PIPELINE_MODEL_INPUT_CONNECTED_TO_MULTIPLE_DATA_SOURCES, "Pipeline definition has multiple connections to the same input of underlying model"},
    {StatusCode::PIPELINE_EXIT_USED_AS_NODE_DEPENDENCY, "Pipeline definition has response node used as dependency node"},
    {StatusCode::PIPELINE_NAME_OCCUPIED, "Pipeline has the same name as model"},

    // Storage errors
    // S3
    {StatusCode::S3_BUCKET_NOT_FOUND, "S3 Bucket not found"},
    {StatusCode::S3_METADATA_FAIL, "S3 metadata failure"},
    {StatusCode::S3_FAILED_LIST_OBJECTS, "S3 Failed to list objects"},
    {StatusCode::S3_FAILED_GET_TIME, "S3 Failed to get modification time"},
    {StatusCode::S3_INVALID_ACCESS, "S3 Invalid access rights"},
    {StatusCode::S3_FILE_NOT_FOUND, "S3 File or directory not found"},
    {StatusCode::S3_FILE_INVALID, "S3 File path is invalid"},
    {StatusCode::S3_FAILED_GET_OBJECT, "S3 Failed to get object from path"},

    // GCS
    {StatusCode::GCS_BUCKET_NOT_FOUND, "GCS Bucket not found"},
    {StatusCode::GCS_METADATA_FAIL, "GCS metadata failure"},
    {StatusCode::GCS_FAILED_LIST_OBJECTS, "GCS Failed to list objects"},
    {StatusCode::GCS_FAILED_GET_TIME, "GCS Failed to list objects"},
    {StatusCode::GCS_INVALID_ACCESS, "GCS Invalid access rights"},
    {StatusCode::GCS_FILE_NOT_FOUND, "GCS File or directory not found"},
    {StatusCode::GCS_FILE_INVALID, "GCS File path is invalid"},
    {StatusCode::GCS_FAILED_GET_OBJECT, "GCS Failed to get object from path"},
    {StatusCode::GCS_INCORRECT_REQUESTED_OBJECT_TYPE, "GCS invalid object type in path"},

    // AS
    {StatusCode::AS_INVALID_PATH, "AS Invalid path"},
    {StatusCode::AS_CONTAINER_NOT_FOUND, "AS Container not found"},
    {StatusCode::AS_SHARE_NOT_FOUND, "AS Share not found"},
    {StatusCode::AS_METADATA_FAIL, "AS metadata failure"},
    {StatusCode::AS_FAILED_LIST_OBJECTS, "AS Failed to list objects"},
    {StatusCode::AS_FAILED_GET_TIME, "AS Failed to list objects"},
    {StatusCode::AS_INVALID_ACCESS, "AS Invalid access rights"},
    {StatusCode::AS_FILE_NOT_FOUND, "AS File or directory not found"},
    {StatusCode::AS_FILE_INVALID, "AS File path is invalid"},
    {StatusCode::AS_FAILED_GET_OBJECT, "AS Failed to get object from path"},
    {StatusCode::AS_INCORRECT_REQUESTED_OBJECT_TYPE, "AS invalid object type in path"},

    // Custom Loader
    {StatusCode::CUSTOM_LOADER_LIBRARY_INVALID, "Custom Loader library not found or cannot open"},
    {StatusCode::CUSTOM_LOADER_LIBRARY_LOAD_FAILED, "Cannot load the custom library"},
    {StatusCode::CUSTOM_LOADER_EXISTS, "The custom loader is already present in loaders list"},
    {StatusCode::CUSTOM_LOADER_NOT_PRESENT, "The custom loader is not present in loaders list"},
    {StatusCode::CUSTOM_LOADER_INIT_FAILED, "Custom Loader LoadInit failed"},
    {StatusCode::CUSTOM_LOADER_ERROR, "Custom Loader Generic / Unknown Error"},
};

const std::map<const StatusCode, grpc::StatusCode> Status::grpcStatusMap = {
    {StatusCode::OK, grpc::StatusCode::OK},

    {StatusCode::PATH_INVALID, grpc::StatusCode::INTERNAL},
    {StatusCode::FILE_INVALID, grpc::StatusCode::INTERNAL},
    {StatusCode::NO_MODEL_VERSION_AVAILABLE, grpc::StatusCode::INTERNAL},
    {StatusCode::NETWORK_NOT_LOADED, grpc::StatusCode::INTERNAL},
    {StatusCode::JSON_INVALID, grpc::StatusCode::INTERNAL},
    {StatusCode::MODELINSTANCE_NOT_FOUND, grpc::StatusCode::INTERNAL},
    {StatusCode::SHAPE_WRONG_FORMAT, grpc::StatusCode::INTERNAL},
    {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, grpc::StatusCode::INTERNAL},
    {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, grpc::StatusCode::INTERNAL},
    {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, grpc::StatusCode::INTERNAL},
    {StatusCode::RESHAPE_ERROR, grpc::StatusCode::FAILED_PRECONDITION},
    {StatusCode::MODEL_MISSING, grpc::StatusCode::NOT_FOUND},
    {StatusCode::MODEL_NAME_MISSING, grpc::StatusCode::NOT_FOUND},
    {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, grpc::StatusCode::NOT_FOUND},
    {StatusCode::MODEL_VERSION_MISSING, grpc::StatusCode::NOT_FOUND},
    {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, grpc::StatusCode::NOT_FOUND},
    {StatusCode::MODEL_VERSION_NOT_LOADED_YET, grpc::StatusCode::NOT_FOUND},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, grpc::StatusCode::NOT_FOUND},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, grpc::StatusCode::NOT_FOUND},
    {StatusCode::MODEL_SPEC_MISSING, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_SIGNATURE_DEF, grpc::StatusCode::INVALID_ARGUMENT},

    // Sequence management
    {StatusCode::SEQUENCE_MISSING, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::SEQUENCE_ALREADY_EXISTS, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::SEQUENCE_ID_NOT_PROVIDED, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_SEQUENCE_CONTROL_INPUT, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::SEQUENCE_ID_BAD_TYPE, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::SEQUENCE_TERMINATED, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE, grpc::StatusCode::INVALID_ARGUMENT},

    // Predict request validation
    {StatusCode::INVALID_NO_OF_INPUTS, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_MISSING_INPUT, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_BATCH_SIZE, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_SHAPE, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_PRECISION, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_VALUE_COUNT, grpc::StatusCode::INVALID_ARGUMENT},
    {StatusCode::INVALID_CONTENT_SIZE, grpc::StatusCode::INVALID_ARGUMENT},

    // Deserialization

    // Should never occur - ModelInstance::validate takes care of that
    {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, grpc::StatusCode::INTERNAL},
    {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, grpc::StatusCode::INTERNAL},

    // Inference
    {StatusCode::OV_INTERNAL_INFERENCE_ERROR, grpc::StatusCode::INTERNAL},

    // Serialization

    // Should never occur - it should be validated during model loading
    {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, grpc::StatusCode::INTERNAL},
    {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, grpc::StatusCode::INTERNAL},

    // GetModelStatus
    {StatusCode::INTERNAL_ERROR, grpc::StatusCode::INTERNAL},
};

const std::map<const StatusCode, net_http::HTTPStatusCode> Status::httpStatusMap = {
    {StatusCode::OK, net_http::HTTPStatusCode::OK},

    // REST handler failure
    {StatusCode::REST_INVALID_URL, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_UNSUPPORTED_METHOD, net_http::HTTPStatusCode::NONE_ACC},

    // REST parser failure
    {StatusCode::REST_BODY_IS_NOT_AN_OBJECT, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_PREDICT_UNKNOWN_ORDER, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_INSTANCES_NOT_AN_ARRAY, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_INPUT_NOT_PREALLOCATED, net_http::HTTPStatusCode::ERROR},
    {StatusCode::REST_NO_INSTANCES_FOUND, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_COULD_NOT_PARSE_INSTANCE, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_INPUTS_NOT_AN_OBJECT, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_NO_INPUTS_FOUND, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_COULD_NOT_PARSE_INPUT, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_PROTO_TO_STRING_ERROR, net_http::HTTPStatusCode::ERROR},
    {StatusCode::REST_UNSUPPORTED_PRECISION, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE, net_http::HTTPStatusCode::ERROR},

    {StatusCode::PATH_INVALID, net_http::HTTPStatusCode::ERROR},
    {StatusCode::FILE_INVALID, net_http::HTTPStatusCode::ERROR},
    {StatusCode::NO_MODEL_VERSION_AVAILABLE, net_http::HTTPStatusCode::ERROR},
    {StatusCode::NETWORK_NOT_LOADED, net_http::HTTPStatusCode::ERROR},
    {StatusCode::JSON_INVALID, net_http::HTTPStatusCode::ERROR},
    {StatusCode::MODELINSTANCE_NOT_FOUND, net_http::HTTPStatusCode::ERROR},
    {StatusCode::SHAPE_WRONG_FORMAT, net_http::HTTPStatusCode::ERROR},
    {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, net_http::HTTPStatusCode::ERROR},
    {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, net_http::HTTPStatusCode::ERROR},
    {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, net_http::HTTPStatusCode::ERROR},
    {StatusCode::RESHAPE_ERROR, net_http::HTTPStatusCode::PRECOND_FAILED},
    {StatusCode::MODEL_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::MODEL_NAME_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::MODEL_VERSION_MISSING, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::MODEL_VERSION_NOT_LOADED_YET, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, net_http::HTTPStatusCode::NOT_FOUND},
    {StatusCode::MODEL_SPEC_MISSING, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_SIGNATURE_DEF, net_http::HTTPStatusCode::BAD_REQUEST},

    // Predict request validation
    {StatusCode::INVALID_NO_OF_INPUTS, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_MISSING_INPUT, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_BATCH_SIZE, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_SHAPE, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_PRECISION, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_VALUE_COUNT, net_http::HTTPStatusCode::BAD_REQUEST},
    {StatusCode::INVALID_CONTENT_SIZE, net_http::HTTPStatusCode::BAD_REQUEST},

    // Deserialization

    // Should never occur - ModelInstance::validate takes care of that
    {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, net_http::HTTPStatusCode::ERROR},
    {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, net_http::HTTPStatusCode::ERROR},

    // Inference
    {StatusCode::OV_INTERNAL_INFERENCE_ERROR, net_http::HTTPStatusCode::ERROR},

    // Serialization

    // Should never occur - it should be validated during model loading
    {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, net_http::HTTPStatusCode::ERROR},
    {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, net_http::HTTPStatusCode::ERROR},

    // GetModelStatus
    {StatusCode::INTERNAL_ERROR, net_http::HTTPStatusCode::ERROR},
};

}  // namespace ovms
