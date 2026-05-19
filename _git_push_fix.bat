@echo on
cd /d C:\AI_Imperium\05_APP\jarvis_api
rmdir /s /q .git
git init
git add .
git commit -m "fix: repair Dockerfile CMD for Railway deploy"
git remote add origin https://github.com/FloScienceboy/jarvis-api.git
git branch -M main
git push -u origin main --force
echo DONE - Railway will now auto-deploy
pause
