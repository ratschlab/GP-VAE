DATA_DIR="data/hmnist"
random_mechanism="mnar"

mkdir -p ${DATA_DIR}

if [ "$random_mechanism" == "mnar" ] ; then
    wget https://www.dropbox.com/s/xzhelx89bzpkkvq/hmnist_mnar.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "spatial"] ; then
    wget https://www.dropbox.com/s/jiix44usv7ibv1z/hmnist_spatial.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "random" ] ; then
    wget https://www.dropbox.com/s/7s5y70f4idw9nei/hmnist_random.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "temporal_neg" ] ; then
    wget https://www.dropbox.com/s/fnqi4rv9wtt2hqo/hmnist_temporal_neg.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
elif [ "$random_mechanism" == "temporal_pos" ] ; then
    wget https://www.dropbox.com/s/tae3rdm9ouaicfb/hmnist_temporal_pos.npz?dl=1 -O ${DATA_DIR}/hmnist_${random_mechanism}.npz
fi
