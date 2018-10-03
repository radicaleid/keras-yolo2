spark-submit --conf spark.shuffle.blockTransferService=nio          \
             --conf spark.scheduler.minRegisteredResourcesRatio=1.0 \
             --conf spark.shuffle.reduceLocality.enabled=false      \
             --conf spark.executor.instances=16                     \
             --conf spark.executor.cores=28                         \
             --conf spark.executor.memory=184g                      \
             --conf spark.driver.memory=184g                        \
             --conf spark.app.name="dense2sparse"                   \
             --total-executor-cores 448                             \
             convert_to_sparse.py
