from dataclasses import dataclass


@dataclass
class CfgSlurm:
	job_id: int = -1
	working_directory: str = "None"
	slurm_user: str = "user_nf"
