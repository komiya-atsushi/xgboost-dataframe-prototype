package jp.gihyo.spark.mllib

import ml.dmlc.xgboost4j.scala.spark.ml.XGBoostBinaryClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexerModel, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object XGBoostClassificationDemo {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("XGBoostClassificationDemo")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val df = sqlContext.createDataFrame(
      sc.textFile("data/SMSSpamCollection").map(_.split("\\s", 2)).map(v => Row(v(0), v(1))),
      StructType(Seq(
        StructField("rawLabel", StringType),
        StructField("text", StringType)
      )))

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")

    val hashingTF = new HashingTF()
      .setNumFeatures(5000)
      .setInputCol("tokens")
      .setOutputCol("features")

    val labelIndexer = new StringIndexerModel(Array("ham", "spam"))
      .setInputCol("rawLabel")
      .setOutputCol("label")

    val xgb = new XGBoostBinaryClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setRound(15)
      .setMaxDepth(6)

    val pipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        hashingTF,
        labelIndexer,
        xgb
      ))

    val paramGrid = new ParamGridBuilder()
      .addGrid(xgb.eta, Array(0.8f, 1.0f))
      .addGrid(xgb.gamma, Array(0.1f, 0.2f, 0.4f))
      .addGrid(xgb.minChildWeight, Array(0.1f, 0.5f, 1.0f))
      .build()

    val cvModel = new CrossValidator()
      .setNumFolds(3)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimator(pipeline)
      .fit(df)

    println(cvModel.avgMetrics.max)

    cvModel.bestModel.parent match {
      case pipeline: Pipeline =>
        pipeline.getStages.zipWithIndex.foreach { case (stage, index) =>
          println(s"Stage[${index + 1}]: ${stage.getClass.getSimpleName}")
          println(stage.extractParamMap())
        }
    }

    val prediction = cvModel.transform(df)

    val colNames = Seq("rawLabel", "label", "rawPrediction", "probability", "prediction")
    prediction
      .select(colNames.map(prediction(_)): _*)
      .show(10, truncate = false)
  }
}
