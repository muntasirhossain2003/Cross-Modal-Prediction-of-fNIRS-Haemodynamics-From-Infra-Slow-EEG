@echo off
cd /d G:\research_bio_medical
"C:\Program Files\Python312\python.exe" -u scripts\run_maisha_all_seeds.py %* > maisha_all_seeds.log 2>&1
