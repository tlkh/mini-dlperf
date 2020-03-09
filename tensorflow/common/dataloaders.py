import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers as xfmers

def return_naive_tfds(dataset_name="imagenette/160px", buffer=8192, num_shards=1, index=0):
    st = time.time()
    
    dataset, info = tfds.load(dataset_name,
                              with_info=True,
                              as_supervised=True)
    num_class = info.features["label"].num_classes
    print("Found", num_class, "classes in", dataset_name)
    num_train = info.splits["train"].num_examples
    num_valid = info.splits["validation"].num_examples
    print("Found training examples:", num_train)
    print("Found validation examples:", num_valid)
    
    train = dataset["train"].shard(num_shards=num_shards, index=index)
    train = train.shuffle(buffer)
    train = train.repeat(count=-1)
    print("Train output:", str(train.take(1)))
    
    valid = dataset["validation"].shard(num_shards=num_shards, index=index)
    valid = valid.repeat(count=-1)
    print("Valid output:", str(valid.take(1)))
    
    dataset = {"train": train,
               "valid": valid,
               "num_train": num_train,
               "num_valid": num_valid,
               "num_class": num_class,}
    
    et = time.time()
    print("Took", int(et-st), "seconds to return dataset")
    
    return dataset

def return_fast_tfds(dataset_name="imagenette/160px", worker_threads=8, buffer=8192, num_shards=1, index=0):
    st = time.time()
    
    options = tf.data.Options()
    read_config = tfds.ReadConfig(options=options, interleave_parallel_reads=worker_threads)
    dataset, info = tfds.load(dataset_name,
                              read_config=read_config,
                              decoders={'image': tfds.decode.SkipDecoding(),},
                              with_info=True,
                              as_supervised=True)
    num_class = info.features["label"].num_classes
    print("Found", num_class, "classes in", dataset_name)
    num_train = info.splits["train"].num_examples
    num_valid = info.splits["validation"].num_examples
    print("Found training examples:", num_train)
    print("Found validation examples:", num_valid)
    
    train = dataset["train"].shard(num_shards=num_shards, index=index)
    train.options().experimental_threading.private_threadpool_size = worker_threads
    train = train.shuffle(buffer)
    train = train.repeat(count=-1)
    print("Train output:", str(train.take(1)))
    
    valid = dataset["validation"].shard(num_shards=num_shards, index=index)
    valid.options().experimental_threading.private_threadpool_size = worker_threads
    valid = valid.repeat(count=-1)
    print("Valid output:", str(valid.take(1)))
    
    dataset = {"train": train,
               "valid": valid,
               "num_train": num_train,
               "num_valid": num_valid,
               "num_class": num_class,}
    
    et = time.time()
    print("Took", int(et-st), "seconds to return dataset")
    
    return dataset
    

def return_glue_task(tokenizer, dataset_name, task_name, max_seq_len=512, index=0, num_shards=1):   
    st = time.time()
    
    data, info = tfds.load(dataset_name, shuffle_files=False,
                           with_info=True)
    
    train_examples = info.splits["train"].num_examples
    valid_examples = info.splits["validation"].num_examples
    test_examples = info.splits["test"].num_examples
    num_labels = info.features["label"].num_classes
    
    print("Task:", dataset_name, ":")
    print("\tTrain:", train_examples)
    print("\tValid:", valid_examples)
    print("\tTest: ", test_examples)

    print("\t[1/3] Converting training dataset...")
    train_dataset = data["train"]
    train_dataset = train_dataset.shard(num_shards, index)
    train_dataset = xfmers.glue_convert_examples_to_features(train_dataset, tokenizer,
                                                             max_length=max_seq_len, task=task_name)

    print("\t[2/3] Converting validation dataset...")
    valid_dataset = data["validation"]
    valid_dataset = valid_dataset.shard(num_shards, index)
    valid_dataset = xfmers.glue_convert_examples_to_features(valid_dataset, tokenizer,
                                                             max_length=max_seq_len, task=task_name)
    
    print("\t[3/3] Converting test dataset...")
    test_dataset = data["validation"]
    test_dataset = xfmers.glue_convert_examples_to_features(test_dataset, tokenizer,
                                                            max_length=max_seq_len, task=task_name)
    
    et = time.time()
    print("Took", int(et-st), "seconds to return dataset")
    
    return {"train_dataset": train_dataset,
            "valid_dataset": valid_dataset,
            "test_dataset": test_dataset,
            "train_examples": train_examples,
            "valid_examples": valid_examples,
            "test_examples": valid_examples, #test_examples,
            "shards": num_shards,
            "num_labels": num_labels}


def create_tokenizer(model_name):
    tokenizer = xfmers.AutoTokenizer.from_pretrained(model_name)
    return tokenizer
