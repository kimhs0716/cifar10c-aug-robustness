@echo off
setlocal

set DATA_DIR=%~1
if "%DATA_DIR%"=="" set DATA_DIR=.\data

echo Downloading CIFAR-10-C to %DATA_DIR% ...
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

where aria2c >nul 2>&1
if %errorlevel%==0 (
    echo Using aria2c ...
    aria2c -x 16 -s 16 -k 1M --user-agent="Wget/1.20.3" ^
        https://zenodo.org/records/2535967/files/CIFAR-10-C.tar ^
        -o "%DATA_DIR%\CIFAR-10-C.tar"
) else (
    echo aria2c not found, using curl ...
    curl -L --progress-bar https://zenodo.org/records/2535967/files/CIFAR-10-C.tar ^
        -o "%DATA_DIR%\CIFAR-10-C.tar"
)

tar -xf "%DATA_DIR%\CIFAR-10-C.tar" -C "%DATA_DIR%\"
del "%DATA_DIR%\CIFAR-10-C.tar"

echo Done. Files saved to %DATA_DIR%\CIFAR-10-C\
endlocal
