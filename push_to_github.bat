@echo off
echo ========================================
echo    PUSH TO GITHUB - SETUP SCRIPT
echo ========================================
echo.
echo This script will help you push your project to GitHub.
echo Make sure you have created a GitHub repository first!
echo.
set /p username="Enter your GitHub username: "
set /p repo="Enter your repository name: "
echo.
echo Setting up remote origin...
git remote add origin https://github.com/%username%/%repo%.git
echo.
echo Renaming branch to main...
git branch -M main
echo.
echo Pushing to GitHub...
git push -u origin main
echo.
echo ========================================
echo             SETUP COMPLETE!
echo ========================================
echo.
echo Check your repository at:
echo https://github.com/%username%/%repo%
echo.
echo If you had authentication issues, you may need to:
echo 1. Create a Personal Access Token on GitHub
echo 2. Use: git remote set-url origin https://%username%:YOUR_TOKEN@github.com/%username%/%repo%.git
echo 3. Run: git push -u origin main
echo.
pause