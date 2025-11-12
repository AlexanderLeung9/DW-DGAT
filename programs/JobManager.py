import os
import sys
import time
import arguments as ag


class JobManager(object):
    __job_schedule_file: str = os.path.join("..", "JobSchedule v2.2.txt")
    __timestamp: str = ""
    __job_name: str = ""
    current_state: str = None

    @staticmethod
    def begin_job(job_name: str):
        assert JobManager.__job_name == "", "A program instance can invoke a job only once."
        assert job_name != "", "You must supply a job name."

        JobManager.__job_name = job_name
        JobManager.__timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        current_dir = os.getcwd()
        project_name = os.path.basename(current_dir)
        JobManager.current_state = "begin"

        with open(JobManager.__job_schedule_file, "a+", encoding="utf-8") as file:
            file.seek(0)
            content = file.read()
            if content == "":
                content = "BeginTime\tProjectName\tJobName\tRootLogDirectory\tJobState\tStateTime\n"
                file.write(content)

            content = f"{JobManager.__timestamp}\t\t{project_name}\t{JobManager.__job_name}\t{ag.Arguments.log_root_dir}\t{JobManager.current_state}\t{JobManager.__timestamp}\n"
            file.write(content)

    @staticmethod
    def update_state(state: str):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        current_dir = os.getcwd()
        project_name = os.path.basename(current_dir)
        JobManager.current_state = state

        with open(JobManager.__job_schedule_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            index = -1
            for i, line in enumerate(lines):
                if JobManager.__timestamp in line and project_name in line and JobManager.__job_name in line:
                    index = i

        if index == -1:
            return

        with open(JobManager.__job_schedule_file, "w", encoding="utf-8") as file:
            line = f"{JobManager.__timestamp}\t\t{project_name}\t{JobManager.__job_name}\t{ag.Arguments.log_root_dir}\t{JobManager.current_state}\t{timestamp}\n"
            lines[index] = line
            content = "".join(lines)
            file.write(content)

    @staticmethod
    def finish_job(exit_code: int):
        if JobManager.__job_name != "":
            JobManager.update_state("finish")
        sys.exit(exit_code)

    @staticmethod
    def shutdown_cloud_server():
        if os.name == "posix" and os.path.expanduser("~") == "/root" and os.path.isdir("/hy-tmp"):
            print("Shutting down the cloud server...")
            os.system("shutdown")
        else:
            print("Not a cloud server. Cancel shutdown.")
