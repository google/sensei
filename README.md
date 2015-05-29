Sensei
======

Sensei is a logistic regression engine. It is intended to be run on one machine
on medium-sized data.

It is currently capable of:

* reading datasets in LIBSVM format (with binary features only)
* highly configurable training
* automated feature exploration
* automated feature pruning
* saving models for later training or scoring
* data scoring

Build
=====

To build Sensei you need to install the following prerequisites:

* C++11 compatibile compiler
* [cmake](http://www.cmake.org/)
* [gflags](https://github.com/gflags/gflags)
* [glog](https://github.com/google/glog)
* [protobuf](https://github.com/google/protobuf)
* [sparsehash](https://code.google.com/p/sparsehash/)

If you use Ubuntu you can install all the prerequisites by running the following
command:
```
sudo apt-get install g++ \
                     cmake \
                     protobuf-compiler \
                     libprotobuf-dev \
                     libgflags-dev \
                     libgoogle-glog-dev \
                     libsparsehash-dev
```

With all the necessary dependencies installed you can build Sensei by running:
```
cmake . && make
```

To build a debug version of the binary, you can run:
```
cmake -DCMAKE_BUILD_TYPE=Debug . && make
```

Test
====
To build Sensei unit tests you need to install the following additional
prerequisites:

* [gtest](https://code.google.com/p/googletest/)

If gtest is installed on your machine, Sensei will use the installed library.
Otherwise, you can put gtest as a subdirectory in Sensei top-level directory.
You can do so by running:
```
wget https://googletest.googlecode.com/files/gtest-1.7.0.zip && unzip gtest-1.7.0.zip
```

To build and run Sensei unit tests run:
```
cmake . && make && ctest
```

Usage
=====

Sensei is configured and controlled by commands defined in `sensei/config.proto`
and `sensei/common_config.proto`. The commands must be provided in text
protobufer format. You can read more about specific command options in comments
in aforementioned files.

To execute Sensei with a specified config run:
```
./sensei_bin --config_files path/to/config_file
```

You can find sample configuration files in the `examples` directory.

Sample config file
------------------

```
command_list {
  command {
    read_data {
      data_reader {
        format: LIBSVM
        training_set {
          files_glob: "input.libsvm"
        }
        feature_spec {
          product {
            prefix: ""
          }
          product {
          }
        }
      }
    }
  }
  command {
    fit_model_weights {
      iterations: 100
    }
  }
  command {
    write_model {
      set {
        format: TEXT
        output_model_path: "model.txt"
      }
    }
  }
  command {
    write_model {
      write {
      }
    }
  }
}
```
