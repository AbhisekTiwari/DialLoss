CUDA_VISIBLE_DEVICES=1     python main.py         --data_path/         --train train.tsv.pre         --valid dev.tsv.pre         --test test.tsv.pre          --save weighted_semantic_context_ce 	    --predict_dir results/         --batch_size 32         --epoch 20         --max_len 350         #--test_only yes         #--continue_train yes \ 