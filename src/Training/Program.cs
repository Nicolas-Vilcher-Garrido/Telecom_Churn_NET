using System.Text.Json;
using System.Linq;
using System.Collections.Generic;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

Console.WriteLine("Training: procurando dados...");

string CleanPath(string relative) => Path.Combine(AppContext.BaseDirectory, relative.Replace('/', Path.DirectorySeparatorChar));
var clean = CleanPath("../../../../../artifacts/clean/telco_clean.csv");
var raw   = CleanPath("../../../../../data/telco.csv");

bool HasRows(string p) => File.Exists(p) && File.ReadLines(p).Skip(1).Any();
bool IsPos(CustomerInput s) =>
    !string.IsNullOrWhiteSpace(s.Churn) && s.Churn.StartsWith("Y", StringComparison.OrdinalIgnoreCase);

var dataSrc = HasRows(clean) ? clean :
              HasRows(raw)   ? raw   :
              throw new InvalidOperationException($"Dataset vazio. clean='{clean}', raw='{raw}'");

Console.WriteLine($"[DBG] dataSrc: {dataSrc}");

var artifactsDir = CleanPath("../../../../../artifacts");
Directory.CreateDirectory(artifactsDir);

var ml = new MLContext(seed: 1);
var data = ml.Data.LoadFromTextFile<CustomerInput>(dataSrc, hasHeader: true, separatorChar: ',');

// ===== Split estratificado para garantir classes no TEST =====
var all = ml.Data.CreateEnumerable<CustomerInput>(data, reuseRowObject: false).ToList();
var pos = all.Where(IsPos).ToList();
var neg = all.Where(x => !IsPos(x)).ToList();
if (pos.Count == 0 || neg.Count == 0)
    throw new InvalidOperationException($"O dataset precisa ter ao menos 1 'Yes' e 1 'No'. Pos={pos.Count}, Neg={neg.Count}");

var rnd = new Random(1);
int takePos = Math.Min(Math.Max(1, pos.Count/5), pos.Count);
int takeNeg = Math.Min(Math.Max(1, neg.Count/5), neg.Count);

var testSet = new List<CustomerInput>();
testSet.AddRange(pos.OrderBy(_ => rnd.Next()).Take(takePos));
testSet.AddRange(neg.OrderBy(_ => rnd.Next()).Take(takeNeg));

var testKeys = new HashSet<string>(testSet.Select(s => s.CustomerID));
var trainSet = all.Where(s => !testKeys.Contains(s.CustomerID)).ToList();

var trainView = ml.Data.LoadFromEnumerable(trainSet);
var testView  = ml.Data.LoadFromEnumerable(testSet);

// ===== Pipeline: CustomMapping (factory) + OHE + Normalize + LightGBM =====
var pipeline =
    ml.Transforms.CustomMapping(new ChurnYesNoToBoolFactory().GetMapping(), contractName: "ChurnYesNoToBool")
    .Append(ml.Transforms.Categorical.OneHotEncoding(new[]
    {
        new InputOutputColumnPair("GenderEncoded","Gender"),
        new InputOutputColumnPair("ContractEncoded","Contract"),
        new InputOutputColumnPair("InternetServiceEncoded","InternetService"),
    }))
    .Append(ml.Transforms.Concatenate("Features",
        "GenderEncoded","ContractEncoded","InternetServiceEncoded",
        "Tenure","MonthlyCharges","TotalCharges"))
    .Append(ml.Transforms.NormalizeMinMax("Features"))
    .Append(ml.BinaryClassification.Trainers.LightGbm(
        new LightGbmBinaryTrainer.Options { LabelColumnName = "LabelBool", FeatureColumnName = "Features" }));

Console.WriteLine("Treinando modelo...");
var model = pipeline.Fit(trainView);

// ===== Avaliação robusta =====
Console.WriteLine("Avaliando...");
BinaryClassificationMetrics metrics;
try
{
    var predTest = model.Transform(testView);
    metrics = ml.BinaryClassification.Evaluate(predTest, labelColumnName: "LabelBool");
}
catch (ArgumentOutOfRangeException)
{
    Console.WriteLine("[WARN] Test set com classe única. Avaliando no conjunto de treino.");
    var predTrain = model.Transform(trainView);
    metrics = ml.BinaryClassification.Evaluate(predTrain, labelColumnName: "LabelBool");
}

Console.WriteLine($"AUC={metrics.AreaUnderRocCurve:0.000}  Accuracy={metrics.Accuracy:0.000}  F1={metrics.F1Score:0.000}");

var modelPath   = Path.Combine(artifactsDir, "model.zip");
var metricsPath = Path.Combine(artifactsDir, "metrics.json");

ml.Model.Save(model, trainView.Schema, modelPath);
await File.WriteAllTextAsync(metricsPath, JsonSerializer.Serialize(new {
    metrics.AreaUnderRocCurve, metrics.Accuracy, metrics.F1Score, Date = DateTime.UtcNow
}, new JsonSerializerOptions { WriteIndented = true }));

Console.WriteLine($"Modelo salvo em {modelPath}\nMétricas em {metricsPath}");
