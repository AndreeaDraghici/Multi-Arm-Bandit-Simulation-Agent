@echo off
pushd %~dp0
pyinstaller agent.spec
popd
pause