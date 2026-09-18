@echo off
rem Double-click this to rebuild games.js and refresh any changed thumbnails.
rem Pass --force to re-shoot every picture:  build.cmd --force
node "%~dp0build.js" %*
echo.
pause
