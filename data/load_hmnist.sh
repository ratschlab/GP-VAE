DATA_DIR="data/hmnist"
random_mechanism="mnar"

mkdir -p ${DATA_DIR}

if [ "$random_mechanism" == "mnar" ] ; then
    wget https://www.dropbox.com/s/aidkzh525mvwf44/hmnist_mnar.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "spatial"] ; then
    wget https://www.dropbox.com/s/ccxlqvu80hk0jfn/hmnist_spatial.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "random" ] ; then
    wget https://www.dropbox.com/s/7iudp0q7fed5map/hmnist_random.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "temporal_neg" ] ; then
    wget https://www.dropbox.com/s/aw2dj0ikd48zf89/hmnist_temporal_neg.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "temporal_pos" ] ; then
    wget https://www.dropbox.com/s/qktos9t0i6i2ee3/hmnist_temporal_pos.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
fi
