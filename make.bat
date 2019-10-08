 @ECHO OFF

pdflatex "main_A"
if %errorlevel% neq 0 exit /b %errorlevel%

bibtex8 "main_A"
if %errorlevel% neq 0 if %errorlevel% neq 1 exit /b %errorlevel%

pdflatex "main_A"
if %errorlevel% neq 0 exit /b %errorlevel%

pdflatex "main_A"
if %errorlevel% neq 0 exit /b %errorlevel%

pdflatex "main_A"
if %errorlevel% neq 0 exit /b %errorlevel%

pdflatex "main_B"
if %errorlevel% neq 0 exit /b %errorlevel%

bibtex8 "main_B"
if %errorlevel% neq 0 if %errorlevel% neq 1 exit /b %errorlevel%

pdflatex "main_B"
if %errorlevel% neq 0 exit /b %errorlevel%

pdflatex "main_B"
if %errorlevel% neq 0 exit /b %errorlevel%

pdflatex "main_B"
if %errorlevel% neq 0 exit /b %errorlevel%
