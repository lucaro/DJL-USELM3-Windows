import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

public class Main {

  public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {

    Criteria<String[], float[][]> criteria =
        Criteria.builder()
            .optApplication(Application.NLP.TEXT_EMBEDDING)
            .setTypes(String[].class, float[][].class)
            .optModelPath(Path.of("models/usem_v3.onnx"))
            .optTranslator(new USEML3Translator())
            .optEngine("OnnxRuntime")
            .optOption("customOpLibrary", "onnxruntime_extensions/ortcustomops.dll")
            .build();

    ZooModel<String[], float[][]> model = criteria.loadModel();

    Predictor<String[], float[][]> predictor = model.newPredictor();

    String[] inputs = {
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding"
    };

    float[][] embeddings = predictor.predict(inputs);

    for (int i = 0; i < inputs.length; ++i) {
      System.out.println("Embedding for: " + inputs[i] + "\n" + Arrays.toString(embeddings[i]));
    }

  }

  private static final class USEML3Translator implements NoBatchifyTranslator<String[], float[][]> {

    USEML3Translator() {
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String[] inputs) {
      NDManager manager = ctx.getNDManager();
      return new NDList(manager.create(inputs));
    }

    @Override
    public float[][] processOutput(TranslatorContext ctx, NDList list) {
      NDList result = new NDList();
      long numOutputs = list.singletonOrThrow().getShape().get(0);
      for (int i = 0; i < numOutputs; i++) {
        result.add(list.singletonOrThrow().get(i));
      }
      return result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
    }
  }

}
