DATA_DIR="data/physionet"

mkdir -p ${DATA_DIR}
wget https://www.dropbox.com/s/651d86winb4cy9n/physionet.npz?dl=1 -O ${DATA_DIR}/physionet.npz
