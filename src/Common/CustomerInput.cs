using Microsoft.ML.Data;

namespace Common;

public class CustomerInput
{
    [LoadColumn(0)]
    public string CustomerID { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string Gender { get; set; } = string.Empty;

    [LoadColumn(2)]
    public float Tenure { get; set; }

    [LoadColumn(3)]
    public float MonthlyCharges { get; set; }

    [LoadColumn(4)]
    public float TotalCharges { get; set; }

    [LoadColumn(5)]
    public string Contract { get; set; } = string.Empty;

    [LoadColumn(6)]
    public string InternetService { get; set; } = string.Empty;

    // Label: "Yes"/"No"
    [LoadColumn(7), ColumnName("Label")]
    public string Churn { get; set; } = string.Empty;
}

public class ChurnPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Predicted { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
