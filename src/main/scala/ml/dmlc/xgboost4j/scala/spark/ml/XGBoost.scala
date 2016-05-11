package ml.dmlc.xgboost4j.scala.spark.ml

import ml.dmlc.xgboost4j.scala.spark.DataUtils
import ml.dmlc.xgboost4j.scala.{Booster => ScalaBooster, DMatrix => ScalaDMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

trait HasRound extends Params {
  final val round: IntParam = new IntParam(this, "round",
    "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  def setRound(value: Int): this.type = set(round, value)
  def getRound: Int = $(round)
}

trait XGBoostGeneralParams extends Params {
  final val booster: Param[String] = new Param(this, "booster",
    "which booster to use, can be gbtree or gblinear. gbtree uses tree based model while gblinear uses linear function.",
    ParamValidators.inArray(Array("gbtree", "gblinear")))

  def setBooster(value: String): this.type = set(booster, value)
  def getBooster: String = $(booster)
  setDefault(booster, "gbtree")
}

trait XGBoostLearner extends HasRound with XGBoostGeneralParams {
  protected def toScalaDMatrix(rdd: RDD[LabeledPoint]): ScalaDMatrix = {
    // TODO 分散処理はひとまず考えないことにする
    new ScalaDMatrix(DataUtils.fromSparkPointsToXGBoostPoints(rdd.collect().iterator))
  }

  protected def buildParams(objective: String): Map[String, Any] = {
    val booster = getBooster
    val boosterParams = booster match {
      case "gbtree" => getTreeBoosterParams
      case "gblinear" => getLinearBoosterParams
    }

    boosterParams + ("objective" -> objective, "booster" -> booster)
  }

  protected def doTrain(labeledPoints: RDD[LabeledPoint], objective: String): ScalaBooster = {
    val trainingSet = toScalaDMatrix(labeledPoints)
    val params = buildParams(objective)
    ScalaXGBoost.train(trainingSet, params, getRound)
  }

  def getTreeBoosterParams: Map[String, Any]
  def getLinearBoosterParams: Map[String, Any]
}

/**
  * Tree Booster (gbtree) が提供しているパラメータ。
  */
trait TreeBoosterParams extends Params {
  final val eta: FloatParam = new FloatParam(this, "eta",
    "step size shrinkage used in update to prevents overfitting",
    ParamValidators.inRange(0.0f, 1.0f, lowerInclusive = true, upperInclusive = true))

  def setEta(value: Float): this.type = set(eta, value)
  def getEta: Float = $(eta)
  setDefault(eta, 0.3f)

  final val gamma: FloatParam = new FloatParam(this, "gamma",
    "minimum loss reduction required to make a further partition on a leaf node of the tree",
    ParamValidators.gtEq(0.0f))

  def setGamma(value: Float): this.type = set(gamma, value)
  def getGamma: Float = $(gamma)
  setDefault(gamma, 0.0f)

  final val maxDepth: IntParam = new IntParam(this, "maxDepth",
    "maximum depth of a tree, increase this value will make model more complex / likely to be overfitting.",
    ParamValidators.gtEq(1))

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)
  def getMaxDepth: Int = $(maxDepth)
  setDefault(maxDepth, 6)

  final val minChildWeight: FloatParam = new FloatParam(this, "minChildWeight",
    "minimum sum of instance weight(hessian) needed in a child.",
    ParamValidators.gtEq(0))

  def setMinChildWeight(value: Float): this.type = set(minChildWeight, value)
  def getMinChildWeight: Float = $(minChildWeight)
  setDefault(minChildWeight, 1.0f)

  // TODO 他にもたくさんパラメータが存在するけど、全部記述するのは大変なので省略

  def getTreeBoosterParams: Map[String, Any] = {
    Map(
      "eta" -> getEta,
      "gamma" -> getGamma,
      "max_depth" -> getMaxDepth,
      "min_child_weight" -> getMinChildWeight
    )
  }
}

/**
  * Linear Booster (gblinear) が提供しているパラメータ。
  */
trait LinearBoosterParams extends Params {
  final val lambda: FloatParam = new FloatParam(this, "lambda",
    "L2 regularization term on weights, increase this value will make model more conservative.")

  def setLambda(value: Float): this.type = set(lambda, value)
  def getLambda: Float = $(lambda)
  setDefault(lambda, 1.0f)

  final val alpha: FloatParam = new FloatParam(this, "alpha",
  "L1 regularization term on weights, increase this value will make model more conservative.")

  def setAlpha(value: Float): this.type = set(alpha, value)
  def getAlpha: Float = $(alpha)
  setDefault(alpha, 0.0f)

  final val lambdaBias: FloatParam = new FloatParam(this, "lambdaBias",
    "L2 regularization term on bias, default 0(no L1 reg on bias because it is not important)")

  def setLambdaBias(value: Float): this.type = set(lambdaBias, value)
  def getLambdaBias: Float = $(lambdaBias)
  setDefault(lambdaBias, 0.0f)

  def getLinearBoosterParams: Map[String, Any] = {
    Map(
      "lambda" -> getLambda,
      "alpha" -> getAlpha,
      "lambda_bias" -> getLambdaBias
    )
  }
}

/**
  * XGBoost による二値分類 (確率あり) の実装例
  */
class XGBoostBinaryClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, XGBoostBinaryClassifier, XGBoostBinaryClassificationModel]
    with XGBoostLearner
    with TreeBoosterParams
    with LinearBoosterParams {

  def this() = this(Identifiable.randomUID("xgb-binCls"))

  override def copy(extra: ParamMap): XGBoostBinaryClassifier = defaultCopy(extra)

  override protected def train(dataset: DataFrame): XGBoostBinaryClassificationModel = {
    val booster = doTrain(extractLabeledPoints(dataset), "binary:logistic")
    new XGBoostBinaryClassificationModel(uid, booster, getRound)
  }
}

class XGBoostBinaryClassificationModel(override val uid: String,
                                       booster: ScalaBooster,
                                       trainedRound: Int)
  extends ProbabilisticClassificationModel[Vector, XGBoostBinaryClassificationModel]
    with HasRound {

  setDefault(round, trainedRound)

  override def copy(extra: ParamMap): XGBoostBinaryClassificationModel = {
    copyValues(new XGBoostBinaryClassificationModel(uid, booster, trainedRound), extra)
  }

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val f = features.toArray.map(x => x.toFloat)
    val matrix = new ScalaDMatrix(f, 1, f.length)
    val prediction = booster.predict(matrix, treeLimit = getRound, outPutMargin = true)
    val r = prediction(0)(0).toDouble
    Vectors.dense(-r, r)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    Vectors.dense(
      sigmoid(rawPrediction(0)),
      sigmoid(rawPrediction(1)))
  }

  protected def sigmoid(value: Double): Double = {
    1.0 / (1.0 + math.exp(-value))
  }
}

/**
  * XGBoost による回帰の実装例
  */
class XGBoostRegressor(override val uid: String)
  extends Predictor[Vector, XGBoostRegressor, XGBoostRegressionModel]
    with XGBoostLearner
    with TreeBoosterParams
    with LinearBoosterParams {

  def this() = this(Identifiable.randomUID("xgb-reg"))

  override def copy(extra: ParamMap): XGBoostRegressor = defaultCopy(extra)

  override protected def train(dataset: DataFrame): XGBoostRegressionModel = {
    val booster = doTrain(extractLabeledPoints(dataset), "reg:linear")
    new XGBoostRegressionModel(uid, booster, getRound)
  }
}

class XGBoostRegressionModel(override val uid: String,
                             booster: ScalaBooster,
                             trainedRound: Int)
  extends PredictionModel[Vector, XGBoostRegressionModel]
    with HasRound {

  override def copy(extra: ParamMap): XGBoostRegressionModel = {
    copyValues(new XGBoostRegressionModel(uid, booster, trainedRound), extra)
  }

  override protected def predict(features: Vector): Double = {
    val f = features.toArray.map(x => x.toFloat)
    val prediction = booster.predict(new ScalaDMatrix(f, 1, f.length), treeLimit = getRound)
    prediction(0)(0).toDouble
  }
}
