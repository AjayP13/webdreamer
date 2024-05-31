#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Pull docker images
mkdir -p docker_images
cd docker_images || exit
wget http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar
wget http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar
wget http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar
wget http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar
wget http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim