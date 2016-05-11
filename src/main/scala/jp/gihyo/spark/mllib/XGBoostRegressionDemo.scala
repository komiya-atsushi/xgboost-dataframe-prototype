package jp.gihyo.spark.mllib

import ml.dmlc.xgboost4j.scala.spark.ml.XGBoostRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object XGBoostRegressionDemo {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("XGBoostRegressionDemo")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val df = sc.textFile("data/housing.data")
      .map(_.trim.split("\\s+").map(_.toDouble))
      .map(v => LabeledPoint(v.last, Vectors.dense(v.take(v.length - 1))))
      .toDF

    val xgb = new XGBoostRegressor()
      .setRound(10)

    val paramGrid = new ParamGridBuilder()
      .addGrid(xgb.eta, Array(0.8f, 1.0f))
      .addGrid(xgb.gamma, Array(0.1f, 0.2f, 0.4f))
      .addGrid(xgb.minChildWeight, Array(0.1f, 0.5f, 1.0f))
      .build()

    val cvModel = new CrossValidator()
      .setEstimator(new Pipeline().setStages(Array(xgb)))
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .fit(df)

    println(cvModel.avgMetrics.min)

    cvModel.bestModel.parent match {
      case pipeline: Pipeline =>
        pipeline.getStages.zipWithIndex.foreach { case (stage, index) =>
          println(s"Stage[${index + 1}]: ${stage.getClass.getSimpleName}")
          println(stage.extractParamMap())
        }
    }
  }
}
