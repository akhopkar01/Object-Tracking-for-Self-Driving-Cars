#!/bin/sh

mkdir -p ../.vscode/

echo \
"{
    \"version\": \"0.2.0\",
    \"configurations\": [
        {
            \"name\": \"g++ - Debug active file\",
            \"type\": \"cppdbg\",
            \"request\": \"launch\",
            \"program\": \""$"{workspaceFolder}/build/app/app\",
            \"args\": [],
            \"stopAtEntry\": true,
            \"cwd\": \""$"{workspaceFolder}\",
            \"environment\": [],
            \"externalConsole\": false,
        }
    ]
}" > ../.vscode/launch.json
