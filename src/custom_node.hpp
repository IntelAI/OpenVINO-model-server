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
#include <unordered_map>

#include "custom_node_interface.hpp"
#include "node.hpp"
#include "nodeinfo.hpp"
#include "pipelineeventqueue.hpp"

namespace ovms {

class NodeLibrary;

class CustomNode : public Node {
    NodeLibrary library;
    parameters_t parameters;
    std::unordered_map<std::string, std::string> nodeOutputNameAlias;

    std::unique_ptr<struct CustomNodeParam[]> _parameters = nullptr;

public:
    CustomNode(
        const std::string& nodeName,
        const NodeLibrary& library,
        const parameters_t& parameters,
        const std::unordered_map<std::string, std::string>& nodeOutputNameAlias = {});

    Status execute(session_key_t sessionKey, PipelineEventQueue& notifyEndQueue) override;

    Status fetchResults(NodeSession& nodeSession, SessionResults& nodeSessionOutputs) override;
    Status fetchResults(BlobMap& outputs, session_key_t sessionKey);

    const std::string& getRealOutputName(const std::string& alias) const {
        return nodeOutputNameAlias.count(alias) == 1 ? nodeOutputNameAlias.at(alias) : alias;
    }

    std::unique_ptr<NodeSession> createNodeSession(const NodeSessionMetadata& metadata, session_id_t shardsCount) override;
};

}  // namespace ovms
