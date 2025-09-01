using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using Common;

Console.WriteLine("Ingestor: lendo data/telco.csv, limpando e salvando em artifacts/clean/telco_clean.csv");

var input = Path.Combine(AppContext.BaseDirectory, "../../../../../data/telco.csv");
var output = Path.Combine(AppContext.BaseDirectory, "../../../../../artifacts/clean/telco_clean.csv");
Directory.CreateDirectory(Path.GetDirectoryName(output)!);

var csvConfig = new CsvConfiguration(CultureInfo.InvariantCulture)
{
    HasHeaderRecord = true,
    DetectDelimiter = true,
    TrimOptions = TrimOptions.Trim,
    BadDataFound = null
};

int ok = 0, bad = 0;

using var reader = new StreamReader(input);
using var csv = new CsvReader(reader, csvConfig);

// >>> ler o cabeçalho antes de GetField("Nome")
if (!await csv.ReadAsync()) 
{
    Console.WriteLine("[WARN] arquivo vazio.");
    Environment.Exit(0);
}
csv.ReadHeader();

using var writer = new StreamWriter(output);
using var outCsv = new CsvWriter(writer, CultureInfo.InvariantCulture);

outCsv.WriteHeader<CustomerInput>();
outCsv.NextRecord();

while (await csv.ReadAsync())
{
    try
    {
        var rec = new CustomerInput
        {
            CustomerID      = (csv.GetField("CustomerID") ?? string.Empty).Trim(),
            Gender          = (csv.GetField("Gender") ?? string.Empty).Trim(),
            Tenure          = ParseFloat(csv.GetField("Tenure"), "Tenure"),
            MonthlyCharges  = ParseFloat(csv.GetField("MonthlyCharges"), "MonthlyCharges"),
            TotalCharges    = ParseFloat(csv.GetField("TotalCharges"), "TotalCharges"),
            Contract        = (csv.GetField("Contract") ?? string.Empty).Trim(),
            InternetService = (csv.GetField("InternetService") ?? string.Empty).Trim(),
            Churn           = NormalizeYesNo(csv.GetField("Churn"))
        };

        if (string.IsNullOrWhiteSpace(rec.CustomerID)) throw new Exception("CustomerID vazio");
        if (rec.Tenure < 0 || rec.Tenure > 120) throw new Exception($"Tenure fora da faixa: {rec.Tenure}");
        if (rec.MonthlyCharges <= 0) throw new Exception("MonthlyCharges inválido");

        outCsv.WriteRecord(rec);
        outCsv.NextRecord();
        ok++;
    }
    catch (Exception ex)
    {
        bad++;
        Console.WriteLine($"[WARN] linha ignorada: {ex.Message}");
    }
}

Console.WriteLine($"Pronto. Registros OK: {ok} | Ignorados: {bad}");
Console.WriteLine($"Arquivo: {output}");

static float ParseFloat(string? raw, string field)
{
    raw = (raw ?? string.Empty).Trim();
    if (string.IsNullOrEmpty(raw)) return 0f;
    if (float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)) return v;
    if (float.TryParse(raw, NumberStyles.Float, new CultureInfo("pt-BR"), out v)) return v;
    throw new Exception($"Campo {field} inválido: '{raw}'");
}

static string NormalizeYesNo(string? raw)
{
    raw = (raw ?? string.Empty).Trim();
    if (string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase)) return "Yes";
    if (string.Equals(raw, "no", StringComparison.OrdinalIgnoreCase)) return "No";
    if (raw == "1") return "Yes";
    if (raw == "0") return "No";
    return string.IsNullOrEmpty(raw) ? "No" : raw;
}
