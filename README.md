


```bash
conda create -n ns3 python=3.7
```

---

## Near-RT RIC

```bash
git clone -b ns-o-ran https://github.com/wineslab/colosseum-near-rt-ric
cd colosseum-near-rt-ric/setup-scripts
./import-wines-images.sh
./setup-ric-bronze.sh
```

```bash
# Terminal 1 logs the E2Term
docker logs e2term -f --since=1s 2>&1 | grep gnb:  # this will help to show only when a gnb is interacting

# Terminal 2 builds and run the x-app container
cd colosseum-near-rt-ric/setup-scripts
./start-xapp-ns-o-ran.sh
```

```bash
cd /home/sample-xapp
./run_xapp.sh
```

## ns-O-RAN

```bash
sudo apt-get update
# Requirements for e2sim
sudo apt-get install -y build-essential git cmake libsctp-dev autoconf automake libtool bison flex libboost-all-dev
# Requirements for ns-3
sudo apt-get install g++ python3
```

```bash
git clone https://github.com/wineslab/ns-o-ran-e2-sim oran-e2sim # this will create a folder called oran-e2sim
cd oran-e2sim/e2sim/
mkdir build
./build_e2sim.sh 3
```

```bash
git clone https://github.com/wineslab/ns-o-ran-ns3-mmwave ns-3-mmwave-oran
cd ns-3-mmwave-oran
```

```bash
cd ns-3-mmwave-oran/contrib
git clone -b master https://github.com/o-ran-sc/sim-ns3-o-ran-e2 oran-interface
cd ..  # go back to the ns-3-mmwave-oran folder
```

---

- https://github.com/tkn-tub/ns3-gym/tree/app


```bash
sudo apt-get update
sudo apt-get install libzmq5 libzmq3-dev
sudo apt-get install libprotobuf-dev
sudo apt-get install protobuf-compiler
```

```bash
cd ./contrib
git clone https://github.com/tkn-tub/ns3-gym.git ./opengym
cd opengym/
git checkout app
```

```bash
conda install protobuf==3.6.1
cp /home/sdin99/ns3/builder.py /home/sdin99/anaconda3/envs/ns3/lib/python3.7/site-packages/google/protobuf/internal/
```


```bash
./waf configure --enable-examples --enable-tests
./waf build
```

```bash
cd ./contrib/opengym/
pip3 install --user ./model/ns3gym
```

Run example:

```bash
cd ./contrib/opengym/examples/opengym/
./simple_test.py
```


---

- https://www.tensorflow.org/install/gpu?hl=ko
- https://developer.nvidia.com/cuda/wsl
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
- https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=WSLUbuntu&target_version=20&target_type=runfilelocal


```bash
conda uninstall protobuf
pip install --upgrade protobuf
conda install tensorflow-gpu==2.1.0
conda install matplotlib
conda install -c conda-forge codecarbon
pip install comet_ml>=3.2.2
```

- tensorflow federated 설치

    ```bash
    pip install --upgrade tensorflow_federated
    pip install nest_asyncio
    pip install matplotlib
    ```

    ```python
    import nest_asyncio
    nest_asyncio.apply()
    ```

<!-- ```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run
``` -->

<!-- ```bash
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
``` -->


<!-- ```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export PATH="/usr/local/cuda/bin:$PATH"
export CUDADIR=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --upgrade pip
pip install tensorflow[and-cuda]==2.10
``` -->

```bash
from tensorflow.python.client import device_lib
import os
# 0: print all messages
# 1: filter out INFO messages
# 2: filter out INFO & WARNING messages
# 3: filter out all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('------------------------------')
print(device_lib.list_local_devices())

```

---

- https://github.com/mlco2/codecarbon
- https://mlco2.github.io/codecarbon/usage.htm

```bash
conda install -c conda-forge codecarbon
```

```bash
codecarbon init
```

- `.codecarbon.config`

    ```bash
    [codecarbon]
    # log_level = DEBUG
    save_to_api = True
    experiment_id = <experiment_id> #the experiment_id you get with init
    ```

```bash
pip install comet_ml>=3.2.2
experiment = Experiment(api_key="YOUR API KEY")
```