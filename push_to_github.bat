@echo off
title Push Face Media Filter to GitHub
echo ========================================================
echo Pushing Face Media Filter Project to GitHub...
echo ========================================================
cd /d "%~dp0"
git push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo SUCCESS! Your project is live on GitHub:
    echo https://github.com/Salmanali9675/face-media-filter
    echo ========================================================
) else (
    echo ========================================================
    echo If repository was not found or failed to push:
    echo 1. Make sure you created 'face-media-filter' on GitHub at:
    echo    https://github.com/new
    echo    (Leave 'Initialize this repository with README' UNCHECKED)
    echo 2. Run this script again!
    echo ========================================================
)
pause
