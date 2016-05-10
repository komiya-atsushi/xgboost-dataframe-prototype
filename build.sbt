name := "xgboost-dataframe-prototype"

version := "1.0"

scalaVersion := "2.10.6"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "ml.dmlc" % "xgboost4j-spark" % "0.5"
)
