#$ -l h_rt=8760:00:00
#$ -o /dev/null
#$ -e /dev/null
#$ -cwd

# Kill all other processes by the user
# shellcheck disable=SC2116
# shellcheck disable=SC2009
ps -aux | grep ajayp | grep singularity | grep -v "$(echo $$)" | awk 'NR!=1 {print $2}' | xargs -I{} kill -9 {}

# Source the user's bashrc
# shellcheck disable=SC1090
source ~/.bashrc

# Change directory to submit location
cd "$SGE_O_WORKDIR/webarena_benchmark_docker_env/docker_images" || exit
export LOGS_DIR="$(pwd)/logs"
rm -rf "$LOGS_DIR"
mkdir -p "$LOGS_DIR"

# Install and setup Singularity
micromamba activate base
micromamba install singularity fakeroot-ng e2fsprogs python==3.11 pip==24.0 -y
export SINGULARITY_CACHEDIR=/dev/shm/ajayp/.singularity
export SINGULARITY_TMPDIR=/dev/shm/ajayp/.singularity
export NON_NFS_DRIVE=/dev/shm/ajayp/.singularity

# Make cache directories
mkdir -p $SINGULARITY_CACHEDIR
mkdir -p $SINGULARITY_TMPDIR
mkdir -p $NON_NFS_DRIVE

# Set hostname
export BORE_HOSTNAME="bore.pub"

# Run Docker images
SANDBOX_IMAGES=("kiwix-serve.sandbox" "shopping_final_0712.sandbox" "shopping_admin_final_0719.sandbox" "postmill-populated-exposed-withimg.sandbox" "gitlab-populated-final-port8023.sandbox")
CONTAINER_NAMES=("wikipedia" "shopping" "shopping_admin" "forum" "gitlab")

read -ra array <<<"$1"
for SGE_TASK_ID in "${array[@]}"; do
    # Main tasks
    if [[ "$SGE_TASK_ID" == "1" ]]; then
        # Setup logs dir
        cp ../webarena_env.txt "$LOGS_DIR"

        # Clean cache directory
        rm -rf $SINGULARITY_CACHEDIR
        rm -rf $SINGULARITY_TMPDIR

        # Make cache directories
        mkdir -p $SINGULARITY_CACHEDIR
        mkdir -p $SINGULARITY_TMPDIR

        # Run Log Server
        (
            echo "<html><head><meta http-equiv=\"refresh\" content=\"0; URL='https://webarena-homepage.ajayp.app'\" /></head>Redirecting to <a href='https://webarena-homepage.ajayp.app'>homepage</a>...</html>" >"$LOGS_DIR/homepage.html"
            (bore local 13949 --to $BORE_HOSTNAME --port 13949) &
            (python3 -m http.server -d "$LOGS_DIR" 13949) &
        ) 1>"$LOGS_DIR/log_server.log" 2>&1

        # Run Homepage
        (
            cd "../../webarena_benchmark/environment_docker/webarena-homepage" || exit
            perl -pi -e "s|http://<your-server-hostname>:3000|http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000|g" templates/index.html
        )
        (
            cd "../../webarena_benchmark/environment_docker/webarena-homepage" || exit
            git checkout app.py
            cat ../../../webarena_benchmark_docker_env/restart.py >>app.py
            pip3 install flask
            (bore local 4399 --to $BORE_HOSTNAME --port 13950) &
            echo "Sleeping for 60 seconds to wait for home page links to update..."
            sleep 60
            (flask run --host=0.0.0.0 --port=4399) &
        ) 1>"$LOGS_DIR/homepage.log" 2>&1 &
    fi

    # Copy everything into the NON NFS DRIVE the first time for those that use it
    if [[ "$SGE_TASK_ID" == "4" ]] || [[ "$SGE_TASK_ID" == "5" ]]; then
        # Only for fakeroot containers
        if [ -f "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run.done" ]; then
            export DISABLE_EXCLUDE=""
        else
            export DISABLE_EXCLUDE="/disable/"
        fi
    else
        export DISABLE_EXCLUDE=""
    fi

    # Run the container
    (
        # Echo hostname
        echo "Hostname: $(hostname)"

        if [[ "$SGE_TASK_ID" == "1" ]]; then
            export ORIG_PORT=80
            export FROM_PORT=8888
            export TO_PORT=13951
            lsof -i tcp:${FROM_PORT} | awk 'NR!=1 {print $2}' | xargs kill 2>/dev/null
            export PUBLIC_HOSTNAME=$(echo "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" | tr _ -)
            (sleep $(($SGE_TASK_ID * 5)) && perl -pi -e "s|http://<your-server-hostname>:$FROM_PORT|https://webarena-$PUBLIC_HOSTNAME.ajayp.app|g" "../../webarena_benchmark/environment_docker/webarena-homepage/templates/index.html") &
            (bore local $FROM_PORT --to $BORE_HOSTNAME --port $TO_PORT) &
            singularity instance stop "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" 2>/dev/null || true
            ###################################################
            # Reset the container
            echo "Resetting the container's data (this can take a few minutes)...: $(date)"
            (
                mkdir -p "${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/"
                singularity --quiet exec --writable --no-home "--bind=$(pwd)/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}/:/source:ro" "--bind=$(pwd)/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/:/target" rsync.sandbox \
                    rsync -a --delete --progress=info2 -O --quiet --stats \
                    --exclude="${DISABLE_EXCLUDE}usr/lib/" \
                    --exclude="${DISABLE_EXCLUDE}usr/share/" \
                    "/source/" \
                    "/target"
            )
            echo "Finished resetting the container's data: $(date)"
            ###################################################
            singularity instance start --writable --no-home "--bind=$(pwd):/data" "${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/" "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i "s/$ORIG_PORT/$FROM_PORT/" /usr/local/bin/start.sh
            (singularity exec --pwd /data "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /.singularity.d/runscript wikipedia_en_all_maxi_2022-05.zim) &
        elif [[ "$SGE_TASK_ID" == "2" ]] || [[ "$SGE_TASK_ID" == "3" ]]; then
            export ORIG_PORT=80
            if [[ "$SGE_TASK_ID" == "2" ]]; then
                export FROM_PORT=7770
                export TO_PORT=13952
            else
                export FROM_PORT=7780
                export TO_PORT=13953
            fi
            lsof -i tcp:${FROM_PORT} | awk 'NR!=1 {print $2}' | xargs kill 2>/dev/null
            export PUBLIC_HOSTNAME=$(echo "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" | tr _ -)
            (sleep $(($SGE_TASK_ID * 5)) && perl -pi -e "s|http://<your-server-hostname>:$FROM_PORT|https://webarena-$PUBLIC_HOSTNAME.ajayp.app|g" "../../webarena_benchmark/environment_docker/webarena-homepage/templates/index.html") &
            (bore local $FROM_PORT --to $BORE_HOSTNAME --port $TO_PORT) &
            singularity instance stop "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" 2>/dev/null || true
            ###################################################
            # Reset the container
            echo "Resetting the container's data (this can take a few minutes)...: $(date)"
            (
                mkdir -p "${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/"
                singularity --quiet exec --writable --no-home "--bind=$(pwd)/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}/:/source:ro" "--bind=$(pwd)/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/:/target" rsync.sandbox \
                    rsync -a --delete --progress=info2 -O --quiet --stats \
                    --exclude="${DISABLE_EXCLUDE}usr/lib/" \
                    --exclude="${DISABLE_EXCLUDE}usr/share/" \
                    --exclude="${DISABLE_EXCLUDE}var/www/magento2/pub/media/catalog/product/" \
                    --exclude="${DISABLE_EXCLUDE}var/www/magento2/vendor" \
                    "/source/" \
                    "/target"
            )
            echo "Finished resetting the container's data: $(date)"g
            ###################################################
            singularity instance start --writable --no-home "${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/" "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i "s/$ORIG_PORT/$FROM_PORT/" /etc/nginx/conf.d/default.conf
            ##################################################
            # echo "Fix permissions..." # For --fakeroot environment
            # singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID-1))]}" chmod -R 777 /var
            # singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID-1))]}" chmod -R 777 /usr
            # singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID-1))]}" chmod -R 777 /tmp/*
            ##################################################
            (singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /.singularity.d/runscript) &
            sleep 60
            echo "Running commands after load."
            if [[ "$SGE_TASK_ID" == "3" ]]; then
                # remove the requirement to reset password (for the admin)
                echo "Admin User & Password: 'admin' with password: 'admin1234'"
                singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
                singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
            fi
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /var/www/magento2/bin/magento setup:store-config:set --base-url="https://webarena-$PUBLIC_HOSTNAME.ajayp.app" # no trailing slash
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='https://webarena-$PUBLIC_HOSTNAME.ajayp.app/' WHERE path = 'web/secure/base_url';"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /var/www/magento2/bin/magento cache:flush
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" curl -s "http://localhost:$FROM_PORT" 1>/dev/null
            echo "Done running commands after load."
        elif [[ "$SGE_TASK_ID" == "4" ]]; then
            export ORIG_PORT=80
            export FROM_PORT=9999
            export TO_PORT=13954
            lsof -i tcp:${FROM_PORT} | awk 'NR!=1 {print $2}' | xargs kill 2>/dev/null
            export PUBLIC_HOSTNAME=$(echo "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" | tr _ -)
            (sleep $(($SGE_TASK_ID * 5)) && perl -pi -e "s|http://<your-server-hostname>:$FROM_PORT|https://webarena-$PUBLIC_HOSTNAME.ajayp.app|g" "../../webarena_benchmark/environment_docker/webarena-homepage/templates/index.html") &
            (bore local $FROM_PORT --to $BORE_HOSTNAME --port $TO_PORT) &
            singularity instance stop "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" 2>/dev/null || true
            ###################################################
            # Reset the container
            echo "Resetting the container's data (this can take a few minutes)...: $(date)"
            (
                mkdir -p "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/"
                singularity --quiet exec --writable --no-home --fakeroot "--bind=$(pwd)/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}/:/source:ro" "--bind=$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/:/target" rsync.sandbox \
                    rsync -a --delete --progress=info2 -O --quiet --stats \
                    --exclude="${DISABLE_EXCLUDE}usr/lib/" \
                    --exclude="${DISABLE_EXCLUDE}usr/share/" \
                    "/source/" \
                    "/target"
                touch "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run.done"
            )
            echo "Finished resetting the container's data: $(date)"
            ###################################################
            singularity instance start --writable --no-home --fakeroot "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/" "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i "s/$ORIG_PORT/$FROM_PORT/" /etc/nginx/conf.d/default.conf
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i "s/9000/9004/" /etc/nginx/conf.d/default.conf
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /bin/bash -c 'echo "listen = 127.0.0.1:9004" >> /usr/local/etc/php-fpm.conf'
            ##################################################
            echo "Fix permissions..." # For --fakeroot environment
            (singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i 's/;user=chrism /user=root/g' /etc/supervisord.conf) &
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod -R 777 /usr
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod -R 777 /run
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chown postgres:postgres /usr/local/pgsql/data
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod 700 /usr/local/pgsql/data
            echo "Done fixing permissions."
            ###################################################
            (singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /.singularity.d/runscript) &
        elif [[ "$SGE_TASK_ID" == "5" ]]; then
            export ORIG_PORT=8023
            export FROM_PORT=8023
            export TO_PORT=13955
            lsof -i tcp:${FROM_PORT} | awk 'NR!=1 {print $2}' | xargs kill 2>/dev/null
            export PUBLIC_HOSTNAME=$(echo "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" | tr _ -)
            (sleep $(($SGE_TASK_ID * 5)) && perl -pi -e "s|http://<your-server-hostname>:$FROM_PORT|https://webarena-$PUBLIC_HOSTNAME.ajayp.app|g" "../../webarena_benchmark/environment_docker/webarena-homepage/templates/index.html") &
            (bore local $FROM_PORT --to $BORE_HOSTNAME --port $TO_PORT) &
            singularity instance stop "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" 2>/dev/null || true
            ###################################################
            # Reset the container
            echo "Resetting the container's data (this can take a few minutes)...: $(date)"
            (
                mkdir -p "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/"
                singularity --quiet exec --writable --no-home --fakeroot "--bind=$(pwd)/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}/:/source:ro" "--bind=$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/:/target" rsync.sandbox \
                    rsync -a --delete --progress=info2 -O --quiet --stats \
                    --exclude="${DISABLE_EXCLUDE}usr/lib/" \
                    --exclude="${DISABLE_EXCLUDE}usr/share/" \
                    "/source/" \
                    "/target"
                touch "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run.done"
            )
            echo "Finished resetting the container's data: $(date)"
            ###################################################
            singularity instance start --writable --no-home --fakeroot "$NON_NFS_DRIVE/${SANDBOX_IMAGES[$(($SGE_TASK_ID - 1))]}.run/" "${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}"
            ##################################################
            echo "Fix permissions..." # For --fakeroot environment
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod -R 777 /opt
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod -R 777 /var
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chown gitlab-psql:gitlab-psql /var/opt/gitlab/postgresql/data
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod 700 /var/opt/gitlab/postgresql/data
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chown gitlab-psql:gitlab-psql /var/opt/gitlab/postgresql/data/server.key
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" chmod 600 /var/opt/gitlab/postgresql/data/server.key
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /bin/bash -c "echo \"gitlab_workhorse['enable'] = true\" >> /etc/gitlab/gitlab.rb"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /bin/bash -c "echo \"gitlab_workhorse['auth_backend'] = 'http://localhost:8080'\" >> /etc/gitlab/gitlab.rb"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /bin/bash -c "echo \"gitlab_workhorse['socket'] = nil\" >> /etc/gitlab/gitlab.rb"
            echo "Done fixing permissions."
            ###################################################
            echo "Running commands before load."
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i "s|^external_url.*|external_url 'https://webarena-$PUBLIC_HOSTNAME.ajayp.app:$FROM_PORT'|" /etc/gitlab/gitlab.rb
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /bin/bash -c "echo \"puma['enable'] = true\" >> /etc/gitlab/gitlab.rb"
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /bin/bash -c "echo \"puma['worker_processes'] = 0\" >> /etc/gitlab/gitlab.rb"
            echo "Done running commands before load."
            singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" sed -i 's/echo "1000000"/# echo "1000000"/g' /opt/gitlab/embedded/bin/runsvdir-start
            (singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" /.singularity.d/runscript /opt/gitlab/embedded/bin/runsvdir-start) &
            sleep 50
            (singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" gitlab-ctl status) &
            # sleep 5
            # (singularity exec "instance://${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}" gitlab-ctl tail puma) &
        fi
    ) 1>"$LOGS_DIR/${CONTAINER_NAMES[$(($SGE_TASK_ID - 1))]}.log" 2>&1 &
done

sleep infinity
