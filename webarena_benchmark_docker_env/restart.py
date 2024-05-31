#####################
# Restart API
#####################
import os

@app.route("/restart")
def restart() -> str:
    # Re-run the new containers by submitting new jobs to the Sun Grid Engine
    os.chdir("../../../")
    os.system("./webarena_benchmark_docker_env/run_docker_images.sh")
    
    # Kill the running jobs on the Sun Grid Engine
    os.system("qstat | grep ajayp | grep bench_ | head -n2 | tac | cut -d' ' -f1 | xargs -I{} qdel {}")

    return "Restarting..."