@echo off
echo ==============================================
echo         Incepe procesul de executie...
echo ==============================================
start multi_arm_agent.exe

rem Verifica daca procesul s-a incheiat cu succes
if %errorlevel% equ 0 (
    echo ============================================
    echo      Procesul s-a incheiat cu succes.
	echo ============================================
) else (
    echo ================================================================
    echo      Procesul a intampinat o problema. Verifica log-urile.
	echo ================================================================
)

rem Verifica daca fisierul executabil exista
if exist "multi_arm_agent.exe" (
	echo =======================================
    echo      Fisierul executabil exista.
	echo =======================================
) else (
	echo ===============================================
    echo      Fisierul executabil nu a fost gasit.
	echo ===============================================
)

pause
