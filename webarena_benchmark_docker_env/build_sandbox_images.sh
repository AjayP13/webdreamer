#$ -l h_rt=60:00:00
#$ -t 1-5
#$ -o build_sandbox_images_$TASK_ID.log
#$ -e build_sandbox_images_$TASK_ID.log
#$ -cwd
#$ -l mem=20G
#$ -pe parallel-onenode 16

# Source the user's bashrc
# shellcheck disable=SC1090
source ~/.bashrc

# Change directory to submit location
cd "$SGE_O_WORKDIR/webarena_benchmark_docker_env/docker_images" || exit

# Install and setup Singularity
micromamba activate base
micromamba install singularity fakeroot-ng -y
export SINGULARITY_CACHEDIR=/dev/shm/ajayp/.singularity
export SINGULARITY_TMPDIR=/dev/shm/ajayp/.singularity

# Clear cache directories
rm -rf $SINGULARITY_CACHEDIR
rm -rf $SINGULARITY_TMPDIR

# Make cache directories
mkdir -p $SINGULARITY_CACHEDIR
mkdir -p $SINGULARITY_TMPDIR

echo "HOST: $(hostname)"
echo "START TIME: $(date)"

# Get rsync container
rm -rf rsync.sandbox
singularity build --sandbox rsync.sandbox docker://ogivuk/rsync:latest

# Build Singularity Image Formats from the Docker Images
DOCKER_IMAGES=("docker://ghcr.io/kiwix/kiwix-serve:3.3.0" "docker-archive://shopping_final_0712.tar" "docker-archive://shopping_admin_final_0719.tar" "docker-archive://postmill-populated-exposed-withimg.tar" "docker-archive://gitlab-populated-final-port8023.tar")
SIF_IMAGES=("kiwix-serve.sandbox" "shopping_final_0712.sandbox" "shopping_admin_final_0719.sandbox" "postmill-populated-exposed-withimg.sandbox" "gitlab-populated-final-port8023.sandbox")
echo "Deleting old copy if exists..."
echo "DELETE START TIME: $(date)"
mkdir -p /tmp/empty
rm -rf /tmp/empty/*
rsync -a --delete /tmp/empty/ "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}"
rm -rf "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}"
echo "DELETE END TIME: $(date)"
echo "Done."
singularity build --sandbox "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}" "${DOCKER_IMAGES[$(($SGE_TASK_ID - 1))]}"
echo "END TIME: $(date)"

# Build copy of sandbox to run & write to
echo "Deleting old copy if exists..."
echo "DELETE START TIME: $(date)"
mkdir -p /tmp/empty
rm -rf /tmp/empty/*
rsync -a --delete /tmp/empty/ "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}.run"
rm -rf "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}.run"
echo "DELETE END TIME: $(date)"
echo "Done."
rsync -a --delete --delete-during --progress=info2 --stats -h "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}/" "${SIF_IMAGES[$(($SGE_TASK_ID - 1))]}.run"
echo "END TIME: $(date)"

# Clear cache directories
rm -rf $SINGULARITY_CACHEDIR
rm -rf $SINGULARITY_TMPDIR
