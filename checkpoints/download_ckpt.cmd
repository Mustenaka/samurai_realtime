@echo off

REM Copyright (c) Meta Platforms, Inc. and affiliates.
REM All rights reserved.

REM This source code is licensed under the license found in the
REM LICENSE file in the root directory of this source tree.

REM Check if wget or curl is available

set CMD=wget

REM Define the URLs for SAM 2.1 checkpoints
set SAM2p1_BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824
set sam2p1_hiera_t_url=%SAM2p1_BASE_URL%/sam2.1_hiera_tiny.pt
set sam2p1_hiera_s_url=%SAM2p1_BASE_URL%/sam2.1_hiera_small.pt
set sam2p1_hiera_b_plus_url=%SAM2p1_BASE_URL%/sam2.1_hiera_base_plus.pt
set sam2p1_hiera_l_url=%SAM2p1_BASE_URL%/sam2.1_hiera_large.pt

REM Download the SAM 2.1 checkpoints
echo Downloading sam2.1_hiera_tiny.pt checkpoint...
%CMD% %sam2p1_hiera_t_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2p1_hiera_t_url%
    exit /b 1
)

echo Downloading sam2.1_hiera_small.pt checkpoint...
%CMD% %sam2p1_hiera_s_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2p1_hiera_s_url%
    exit /b 1
)

echo Downloading sam2.1_hiera_base_plus.pt checkpoint...
%CMD% %sam2p1_hiera_b_plus_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2p1_hiera_b_plus_url%
    exit /b 1
)

echo Downloading sam2.1_hiera_large.pt checkpoint...
%CMD% %sam2p1_hiera_l_url%
if %errorlevel% neq 0 (
    echo Failed to download checkpoint from %sam2p1_hiera_l_url%
    exit /b 1
)

echo All checkpoints are downloaded successfully.
