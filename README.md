# nlp-tools

This project contains the C++ source code for a general-purpose part-of-speech (POS) tagger using the lookahead tagging algorithm described in [[1]](www.aclweb.org/anthology/W11-0328.pdf).

This source is based on the reference implementation provided by the paper authors (http://www.logos.t.u-tokyo.ac.jp/~tsuruoka/lapos/), originally under MIT license. 


## Building

This project uses CMake. On GNU/Linux, use the following commands to compile:

    % mkdir build
    % cd build
    % cmake ..  

## Training


You can build a tagging model using your own annotated corpus. Use the `tool_train` command:

    % ./tool_train -m model.la -i tagged-training.txt 


## Predicting

Once you have trained the model, you can use it by specifying the directory that contains the generated model file.

    % ./tool_predict -m . < untagged-test.txt > tmp

Note that the input must be one-sentence-per-line, and the words need to be already tokenized with white spaces.


## Evaluating

You can evaluate tagging accuracy by the `tool_evaluate` command.

    % ./tool_evaluate tagged-test.txt tmp 


## References

      [1] Yoshimasa Tsuruoka, Yusuke Miyao and Jun'ichi Kazama;
      Learning with Lookahead: Can History-based Models Rival Globally Optimized Models?
      In Proceedings of the Fifteenth Conference on Computational Natural Language Learning (CoNLL), 
      pp. 238-246, 2011.

## License

Except where explicitly indicated otherwise, all source codes of this project are provide under Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
