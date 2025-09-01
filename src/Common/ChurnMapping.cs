using System;
using Microsoft.ML.Transforms;

namespace Common;

// Classe de saída para o label booleano
public class LabelOut
{
    public bool LabelBool { get; set; }
}

// Factory registrada com o mesmo contrato usado no treinamento
[CustomMappingFactoryAttribute("ChurnYesNoToBool")]
public class ChurnYesNoToBoolFactory : CustomMappingFactory<CustomerInput, LabelOut>
{
    public override Action<CustomerInput, LabelOut> GetMapping() =>
        (src, dst) =>
        {
            dst.LabelBool = !string.IsNullOrWhiteSpace(src.Churn)
                            && src.Churn.StartsWith("Y", StringComparison.OrdinalIgnoreCase);
        };
}
