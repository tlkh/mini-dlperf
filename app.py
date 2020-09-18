import streamlit as st
import time
import multiprocessing
import pandas as pd
from common import utils

st.sidebar.title("Mini-DLPerf")
st.sidebar.subheader("\nControls")
threads = st.sidebar.number_input("Threads", min_value=1, value=multiprocessing.cpu_count()-2)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=64)
ready = st.sidebar.checkbox("Ready to run!")

_ = utils.run_command("nvidia-smi nvlink -sc 0bz")

with st.spinner("Getting GPU info..."):
    @st.cache
    def app_get_gpu_info():
        return utils.get_gpu_info()
    st.markdown("GPU info:")
    st.json(app_get_gpu_info())

threads = str(threads)
batch_size = str(batch_size)

if ready:
    progress_bar = st.progress(0)

    start_time = time.time()

    exp_name = "rn50"
    with st.spinner("Running: "+exp_name):
        progress_bar.progress(1)
        results = utils.run_command("python3 run_cnn.py --threads "+threads+" --batch_size "+batch_size)
        if results[-1].split(",")[0] == "PASS":
            rn50_10gb = exp_name + "," + results[-1]
            st.success(rn50_10gb)
        else:
            st.error(exp_name, "FAIL")
        progress_bar.progress(33)

    exp_name = "rn50_imgaug"
    with st.spinner("Running: "+exp_name):
        progress_bar.progress(34)
        results = utils.run_command("python3 run_cnn.py --threads "+threads+" --img_aug --batch_size "+batch_size)
        if results[-1].split(",")[0] == "PASS":
            rn50_imgaug_10gb = exp_name + "," + results[-1]
            st.success(rn50_imgaug_10gb)
        else:
            st.error(exp_name, "FAIL")
        progress_bar.progress(66)

    exp_name = "hugecnn-10gb"
    with st.spinner("Running: "+exp_name):
        progress_bar.progress(67)
        results = utils.run_command("python3 run_cnn.py --threads "+threads+" --huge_cnn --batch_size "+batch_size)
        if results[-1].split(",")[0] == "PASS":
            hugecnn_10gb = exp_name + "," + results[-1]
            st.success(hugecnn_10gb)
        else:
            st.error(exp_name, "FAIL")
        progress_bar.progress(99)

    end_time = time.time()

    progress_bar.progress(100)

    st.text("Total time taken:", int(end_time-start_time), "seconds.")

    cols = ["name", "passed", "avg_fps", "avg_sm_%", "avg_mem_io_%", "avg_pcie_%", "pcie_gbps", "avg_pwr_%", "pwr_watts", "avg_temp", "max_vram", "avg_nvlink", "throttle"]
    df = pd.DataFrame([rn50_10gb.split(","), rn50_imgaug_10gb.split(","), hugecnn_10gb.split(",")], 
                    columns=cols) 

    st.dataframe(df)

    def get_table_download_link(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

else:
    st.markdown("Enter the options in the sidebar and tick 'Ready' to start the benchmark.")
