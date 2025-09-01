using System.Text.Json;
using Common;
using Microsoft.ML;
using Serilog;
using Microsoft.OpenApi.Models;


var builder = WebApplication.CreateBuilder(args);

// Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "Churn Scoring API",
        Version = "v1",
        Description = "API para pontuação de churn (ML.NET)"
    });
});

// Serilog básico
Log.Logger = new LoggerConfiguration()
    .Enrich.FromLogContext()
    .WriteTo.Console()
    .CreateLogger();
builder.Host.UseSerilog();

var app = builder.Build();

// Swagger UI
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Churn Scoring API v1");
    c.RoutePrefix = "swagger"; // UI estará em /swagger
});

// Caminhos utilitários
string CleanPath(string relative) =>
    Path.Combine(AppContext.BaseDirectory, relative.Replace('/', Path.DirectorySeparatorChar));

var modelPath   = CleanPath("../../../../../artifacts/model.zip");
var metricsPath = CleanPath("../../../../../artifacts/metrics.json");

// MLContext e registro dos componentes usados no treinamento
var ml = new MLContext();
// O CustomMapping com contrato "ChurnYesNoToBool" está em Common (ChurnYesNoToBoolFactory)
ml.ComponentCatalog.RegisterAssembly(typeof(ChurnYesNoToBoolFactory).Assembly);

PredictionEngine<CustomerInput, ChurnPrediction>? engine = null;
if (File.Exists(modelPath))
{
    var model = ml.Model.Load(modelPath, out _);
    engine = ml.Model.CreatePredictionEngine<CustomerInput, ChurnPrediction>(model);
    app.Logger.LogInformation("Modelo carregado de {ModelPath}", modelPath);
}
else
{
    app.Logger.LogWarning("Modelo não encontrado em {ModelPath}. Rode o projeto Training primeiro.", modelPath);
}

// Endpoints básicos
app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

app.MapGet("/", () => Results.Text("Churn Scoring API (.NET 8) - use /swagger, /demo ou POST /score"));

// Endpoint de scoring (visível no Swagger)
app.MapPost("/score", (ScoreRequest req) =>
{
    if (engine is null) return Results.Problem("Modelo não carregado. Treine antes (src/Training).");

    var input = new CustomerInput
    {
        CustomerID = req.CustomerID ?? Guid.NewGuid().ToString("N"),
        Gender = req.Gender ?? "Female",
        Tenure = req.Tenure,
        MonthlyCharges = req.MonthlyCharges,
        TotalCharges = req.TotalCharges,
        Contract = req.Contract ?? "Month-to-month",
        InternetService = req.InternetService ?? "Fiber optic",
        Churn = "No" // não usado no scoring
    };

    var pred = engine.Predict(input);
    return Results.Ok(new { pred.Predicted, pred.Probability, pred.Score });
})
.WithName("Score");

// Info do modelo (existe? métricas?)
app.MapGet("/model/info", () =>
{
    object? metricsObj = null;
    if (File.Exists(metricsPath))
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(metricsPath));
            metricsObj = doc.RootElement.Clone();
        }
        catch { /* ignora parse error */ }
    }

    return Results.Ok(new
    {
        modelExists = File.Exists(modelPath),
        modelPath,
        metrics = metricsObj
    });
});

// Demo HTML simples pra testar via navegador
app.MapGet("/demo", () => Results.Content(@"
<!doctype html><meta charset='utf-8'>
<title>Telco Churn – Demo</title>
<style>
  body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 12px}
  label{display:block;margin:8px 0}
  input{padding:6px 8px;width:320px}
  button{padding:8px 14px;margin-top:12px;cursor:pointer}
  pre{background:#111;color:#0f0;padding:12px;border-radius:8px;white-space:pre-wrap}
</style>
<h1>Telco Churn – Demo</h1>
<form onsubmit='send(event)'>
  <label>CustomerID <input id='CustomerID' value='C9999'></label>
  <label>Gender <input id='Gender' value='Female'></label>
  <label>Tenure <input id='Tenure' type='number' value='3'></label>
  <label>MonthlyCharges <input id='MonthlyCharges' type='number' value='120'></label>
  <label>TotalCharges <input id='TotalCharges' type='number' value='360'></label>
  <label>Contract <input id='Contract' value='Month-to-month'></label>
  <label>InternetService <input id='InternetService' value='Fiber optic'></label>
  <button>Score</button>
</form>
<p><a href='/swagger'>Abrir Swagger</a></p>
<pre id='out'></pre>
<script>
async function send(e){
  e.preventDefault();
  const body = {
    CustomerID: CustomerID.value,
    Gender: Gender.value,
    Tenure: parseFloat(Tenure.value),
    MonthlyCharges: parseFloat(MonthlyCharges.value),
    TotalCharges: parseFloat(TotalCharges.value),
    Contract: Contract.value,
    InternetService: InternetService.value
  };
  const r = await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  document.getElementById('out').textContent = await r.text();
}
</script>
","text/html"));

app.Run();

// DTO para o POST /score
public record ScoreRequest(
    string? CustomerID,
    string? Gender,
    float Tenure,
    float MonthlyCharges,
    float TotalCharges,
    string? Contract,
    string? InternetService
);
