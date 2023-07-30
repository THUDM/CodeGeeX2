![](resources/codegeex_logo.png)

<p align="center">
    🏠 <a href="https://codegeex.cn" target="_blank">Homepage</a>｜🛠 Extensions <a href="https://marketplace.visualstudio.com/items?itemName=aminer.codegeex" target="_blank">VS Code</a>, <a href="https://plugins.jetbrains.com/plugin/20587-codegeex" target="_blank">Jetbrains</a>｜🤗 <a href="https://huggingface.co/THUDM/codegeex2-6b" target="_blank">HF Repo</a>｜📄 <a href="https://arxiv.org/abs/2303.17568" target="_blank">Paper</a>
</p>

<p align="center">
    👋 Rejoignez nous sur <a href="https://discord.gg/8gjHdkmAN6" target="_blank">Discord</a>, <a href="https://join.slack.com/t/codegeexworkspace/shared_invite/zt-1s118ffrp-mpKKhQD0tKBmzNZVCyEZLw" target="_blank">Slack</a>, <a href="https://t.me/+IipIayJ32B1jOTg1" target="_blank">Telegram</a>, <a href="resources/wechat.md"target="_blank">WeChat</a>
</p>

查看[中文版](README.md)<br>
Read this in [English](README_EN.md)<br>
[日本語](README_JA.md)で読む

# CodeGeeX2: Un Modèle de Génération de Code Plus Puissant

CodeGeeX2 est la deuxième itération du modèle de génération de code multilingue [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)), basé sur [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) et entrainé sur un large corpus de code. Grâce à l'architecture ChatGLM2, CodeGeeX2 excelle sur une multitude de tâches de génération de code (+107% > CodeGeeX; avec seulement 6 milliards de paramètres, dépassant StarCoder-15B pour certaines tâches). CodeGeeX2 possède les fonctionnalités suivantes:

* **Capacités de Génération de Code Accrues**: Basé sur ChatGLM2-6B, CodeGeeX2-6B à été entrainé sur un dataset de 600 milliards de tokens de plus ce qui a propulsé ses capacités de génération de code par rapport à la génération précédente. Sur [HumanEval-X](https://huggingface.co/datasets/THUDM/humaneval-x), le modèle opère bien mieux que son prédécesseur (Python +57%, C++ +71%, Java +54%, JavaScript +83%, Go +56%, Rust +321\%). En Python, CodeGeeX atteint un score Pass@1 de 35.9%, surpassant StarCoder-15B malgré le fait que CodeGeeX ait ~3 fois moins de paramètres.
* **Des Fonctionnalités Plus Utiles**: Héritant des fonctionnalités de ChatGLM2-6B, CodeGeeX2-6B prend mieux en charge les prompts en chinois et en anglais, peut ingérer jusqu'à 8192 tokens, et se dotte d'une vitesse de génération en inference fortement accrue comparé à la dernière génération. Après quantisation, CodeGeeX fonctionne sur un GPU avec >6GB de mémoire, permettant un déploiement local efficace.
* **Un Assistant Intelligent dans votre Éditeur**: Les plugins ([VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex), et [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)) ont été mis à jour et sont maintenant compatible avec plus de 100 langages de programmation. Le modèle, couplé à l'extension, permet désormais aux utilisateurs de générer du code pour plusieurs fichiers ainsi que de générer et modifier des sections de code. CodeGeeX2 est maintenant capable de résoudre de nombreux problèmes de programmation. Les utilisateurs peuvent profiter de la fonctionnalité "Ask CodeGeeX" pour discuter de manière interactive avec un AI-assistant afin de résumer et d'expliquer du code, traduire du code entre langages, rajouter des commentaires, etc. CodeGeeX permet de maximiser la productivité de ses utilisateurs.
* **License Open-Source**: Les poids du modèle CodeGeeX2-6B sont en accès libre pour toute utilisation dans le cadre de la recherche. Pour toute utilisation commerciale, merci de consulter ce [formulaire](https://open.bigmodel.cn/mla/form?mcode=CodeGeeX2-6B).


## Assistant Intelligent

![](resources/codegeex_demo.png)

Nous avons développé une extension pour VS Code, IntelliJ IDEA, PyCharm, GoLand, WebStorm, and Android Studio. L'extension permet de profiter des capacités du modèle CodeGeeX2 et de générer, annoter et traduire du code. La fonctionnalité "Ask CodeGeeX" permet de coder de manière interactive et améliore grandement votre productivité. Téléchargez l'extension CodeGeeX dans votre IDE pour une meilleure expérience de développement. Trouvez plus de détail sur notre [site]( https://codegeex.cn/).

## Utilisation

Pour exécuter [CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b), utilisez la librairie `transformers`：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True, device='cuda')
model = model.eval()

# TIP: Utilisez un tag pour identifier le langage dans lequel vous souhaitez générer.
prompt = "# language: Python\n# write a bubble sort function\n"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_length=256, top_k=1)
response = tokenizer.decode(outputs[0])

>>> print(response)
# language: Python
# write a bubble sort function


def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
    return list


print(bubble_sort([5, 2, 1, 8, 4]))
```

Accéder à la démo Gradio:
```
python ./demo/run_demo.py
```

❗️Attention:
* Cette version de CodeGeeX2 est capable de compléter / expliquer / traduire du code mais n'a pas été fine-tuned pour être utilisé comme un chatbot. Pour accéder à la version chatbot de CodeGeeX, utilisez les extensions [VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex) et [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex).
* Pour controller le langage dans lequel CodeGeeX2 opère, utilisez des tags formattés ainsi: `# language: Python`. La liste de tous les langages de programmations que CodeGeeX supporte est accessible [ici](https://github.com/THUDM/CodeGeeX2/blob/main/evaluation/utils.py#L14).
* Si vous avez besoin d'utiliser plusieurs GPU pour charger le modèle, vous pouvez utiliser le code suivant:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True, device='cuda')
    model = model.eval()
    ```
    Remplacer par

    ```python
    def get_model():
        tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
        from gpus import load_model_on_gpus
        # Le fichier "gpus" se trouve dans le dossier de démonstration
        model = load_model_on_gpus("THUDM/codegeex2-6b", num_gpus=2)
        model = model.eval()
        return tokenizer, model

    tokenizer, model = get_model()
    ```

## Evaluation

CodeGeeX2 est un modèle de base capable de générer du code en plusieurs langages de programmation et qui est bien plus performant que la version précédente. Voici les capacités de CodeGeeX sur les benchmarks HumanEval, HumanEval-X, et DS1000 (la métrique Pass@k est la même que celle décrite dans ce [papier](https://arxiv.org/abs/2303.17568)):

### HumanEval (Pass@1,10,100)

| **Model**           | **Pass@1** | **Pass@10** | **Pass@100** |
| :-----------------: | :--------: | :---------: | :----------: |
| CodeGen-16B-multi   | 19\.2      | 34\.6       | 55\.2        |
| CodeGeeX-13B        | 22\.9      | 39\.6       | 60\.9        |
| Codex-12B           | 28\.8      | 46\.8       | 72\.3        |
| CodeT5Plus-16B-mono | 30\.9      | 51\.6       | 76\.7        |
| Code-Cushman-001    | 33\.5      | 54\.3       | 77\.4        |
| LLaMA-65B           | 23\.7      | -           | 79\.3        |
| LLaMA2-70B          | 29\.9      | -           | -            |
| CodeGen2\.5-7B-mono | 33\.4      | 58\.4       | 82\.7        |
| StarCoder-15B       | 33\.2      | 61\.0       | 84\.7        |
| **CodeGeeX2-6B**    | **35\.9**  | **62\.6**   | **88\.3**    |
> `n=20, t=0.2, top_p=0.95` pour **Pass@1**; `n=200, t=0.8, top_p=0.95` pour **Pass@10** et **Pass@100**.

### HumanEval-X (Pass@1)

| **Model**                | **Python** | **C++**   | **Java**  | **JavaScript** | **Go**    | **Rust**  | **Overall** |
| :------------------: | :--------: | :-------: | :-------: | :------------: | :-------: | :-------: | :---------: |
| CodeGen-16B-multi    | 19\.2      | 18\.1     | 15\.0     | 18\.4          | 13\.0     | 1\.8      | 14\.2       |
| CodeGeeX-13B         | 22\.9      | 17\.1     | 20\.0     | 17\.6          | 14\.4     | 4\.3      | 16\.0       |
| Replit-code-v1-3B    | 22\.0      | 20\.1     | 20\.1     | 20\.1          | 12\.2     | 8\.6      | 17\.2       |
| CodeGen2\.5-7B-multi | 30\.6      | 24\.3     | 29\.0     | 27\.5          | 18\.9     | **20\.1** | 25\.1       |
| StarCoder-15B        | 35\.5      | 28\.2     | **31\.5** | **33\.2**      | 21\.3     | 17\.8     | 27\.9       |
| **CodeGeeX2-6B**         | **35\.9**  | **29\.3** | 30\.8     | 32\.2          | **22\.5** | 18\.1     | **28\.1**   |
> `n=20, t=0.2, top_p=0.95` for **Pass@1**.

Les résultats ci-dessus peuvent être reproduits avec le script `scripts/run_humanevalx.sh`. Les environements utilisés sont renseignés [ici](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/benchmark/README_zh.md).

### DS1000 (Pass@1)

| **Model**            | **Matplotlib** | **Numpy** | **Pandas** | **Pytorch** | **SciPy** | **Scikit-learn** | **TensorFlow** | **Overall** |
| :--------------: | :------------: | :-------: | :--------: | :---------: | :-------: | :--------------: | :------------: | :---------: |
| \# Samples       | 155            | 220       | 291        | 68          | 106       | 115              | 45             | 1000        |
| CodeGen-16B-Mono | 31\.7          | 10\.9     | 3\.4       | 7\.0        | 9\.0      | 10\.8            | 15\.2          | 11\.7       |
| code-cushman-001 | 40\.7          | 21\.8     | 7\.9       | 12\.4       | 11\.3     | 18\.0            | 12\.2          | 18\.1       |
| Codex-001        | 41\.8          | 26\.6     | 9\.4       | 9\.7        | 15\.0     | 18\.5            | 17\.2          | 20\.2       |
| **CodeGeeX2-6B** | 40\.5          | 25\.5     | 14\.5      | 17\.3       | 19\.3     | 24\.0            | 23\.0          | 23\.1       |
| StarCoder-15B    | 51\.7          | 29\.7     | 11\.4      | 21\.4       | 20\.2     | 29\.5            | 24\.5          | 26\.0       |
| Codex-002        | **57\.0**      | **43\.1** | **26\.5**  | **41\.8**   | **31\.8** | **44\.8**        | **39\.3**      | **39\.2**   |
> `n=40, t=0.2, top_p=0.5` for **Pass@1**。

Les résultats ci-dessus peuvent être reproduits avec le code présent sur le repository [HKUNLP/DS-1000](https://github.com/HKUNLP/DS-1000.git).

## Inference

CodeGeeX2 est bien plus simple à déployer que la génération précédente. L'utilisation de "Multi-Query Attention" et "Flash Attention" accélère grandement la vitesse de génération et le modèle n'a besoin que de 6GB de mémoire après avoir été quantisé en INT4.

### Quantisation

| **Model**        | FP16/BF16 | INT8    | INT4   |
| :--------------: | :-------: | :-----: | :----: |
| CodeGeeX-13B     | 26\.9 GB   | 14\.7 GB | -      |
| **CodeGeeX2-6B** | 13\.1 GB  | 8\.2 GB  | 5\.5 GB |
> Résultats obtenus avec PyTorch 2.0, avec `torch.nn.functional.scaled_dot_product_attention` qui est une version plus rapide du calcul de l'attention.

### Accelération

| **Model**        | **Inference speed (token/s)** |
| :--------------: | :-------------: |
| CodeGeeX-13B     | 32              |
| **CodeGeeX2-6B** | 94              |
> `batch_size=1, max_length=2048` et en utilisant l'accélération des GPUs `GeForce RTX-3090`。

## License

Le code dans ce dépôt est en libre accès selon les droits et devoirs prévu par la license [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). Les poids du modèle sont régis par la [license du modèle](MODEL_LICENSE). Les poids du modèle CodeGeeX2-6B sont en accès libre pour toute utilisation dans le cadre de la recherche. Pour toute utilisation commerciale, merci de consulter ce [formulaire](https://open.bigmodel.cn/mla/form?mcode=CodeGeeX2-6B).


## Citation

Si vous trouvez ce projet utile, n'hésitez pas à citer notre papier:

```
@inproceedings{zheng2023codegeex,
      title={CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X},
      author={Qinkai Zheng and Xiao Xia and Xu Zou and Yuxiao Dong and Shan Wang and Yufei Xue and Zihan Wang and Lei Shen and Andi Wang and Yang Li and Teng Su and Zhilin Yang and Jie Tang},
      booktitle={KDD},
      year={2023}
}
```
