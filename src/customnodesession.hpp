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

#include "blobmap.hpp"
#include "custom_node_interface.hpp"
#include "nodesession.hpp"
#include "pipelineeventqueue.hpp"
#include "status.hpp"

namespace ovms {

class ModelManager;
class Node;
class NodeLibrary;

class CustomNodeSession : public NodeSession {
    const NodeLibrary& library;
    BlobMap resultBlobs;

public:
    CustomNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const NodeLibrary& library, session_id_t shardsCount);
    CustomNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, const NodeLibrary& library, session_id_t shardsCount);
    virtual ~CustomNodeSession();

    Status execute(
        PipelineEventQueue& notifyEndQueue,
        Node& node,
        const NodeLibrary& library,
        std::unique_ptr<struct CustomNodeParam[]>& parameters,
        int parametersLength);

    Status fetchResult(const std::string& name, InferenceEngine::Blob::Ptr& resultBlob);

    Status createBlob(const struct CustomNodeTensor* tensor, InferenceEngine::Blob::Ptr& resultBlob);

    void clearInputs();

    void release() override;
};
}  // namespace ovms
