import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import random
from typing import Dict, List, Tuple, Any, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. ГЕНЕРАЦИЯ ДАТАСЕТА НПА

class NPADatasetGenerator:
    """генератор датасета НПА"""

    # Предопределенные константы
    DOC_TYPES = [
        ('federal_law', 'Федеральный закон', 0.25),
        ('government_decree', 'Постановление Правительства', 0.20),
        ('ministry_order', 'Приказ министерства', 0.15),
        ('fstek_document', 'Документ ФСТЭК', 0.15),
        ('fsb_document', 'Документ ФСБ', 0.10),
        ('gost', 'ГОСТ', 0.10),
        ('methodology', 'Методические рекомендации', 0.05)
    ]

    DOMAINS = [
        ('L02.1', 'Персональные данные', ['152-ФЗ', 'ФЗ-152']),
        ('L00.6', 'Угрозы ИБ', ['ФСТЭК', 'модель угроз']),
        ('L00.8', 'Контроль и надзор', ['Роскомнадзор', 'контроль']),
        ('L01.2.2', 'Коммерческая тайна', ['98-ФЗ', 'коммерческая тайна']),
        ('L00.1', 'Информационная безопасность', ['149-ФЗ', 'ИБ']),
        ('L00.3', 'Государственные системы', ['ГосТех', 'госсистемы']),
        ('L00.7', 'Управление рисками', ['риски', 'Банк России']),
        ('L01.1', 'Защита информации', ['защита', 'шифрование'])
    ]

    TEXT_KEYWORDS = [
        "обеспечение безопасности", "защита информации", "требования к системе",
        "организационные меры", "технические средства", "контроль доступа",
        "аудит безопасности", "управление рисками", "обработка данных",
        "хранение информации", "передача данных", "шифрование",
        "аутентификация", "авторизация", "мониторинг", "отчетность"
    ]

    THEMES = ['безопасность', 'защита', 'конфиденциальность', 'целостность',
              'доступность', 'аудит', 'контроль', 'шифрование', 'аутентификация']

    MINISTRIES = ['Минцифры', 'Минобрнауки', 'Минэкономразвития', 'Минтруд']

    def __init__(self, num_documents=50):
        self.num_documents = num_documents
        self.documents = []
        self.graph = nx.Graph()

    def _get_random_type(self):
        """Вероятностный выбор типа документа"""
        types, labels, probs = zip(*self.DOC_TYPES)
        return np.random.choice(types, p=probs), labels[types.index(np.random.choice(types, p=probs))]

    def generate_document(self, doc_id):
        """генерация документа"""
        # Выбор типа и домена
        doc_type, doc_label = self._get_random_type()
        domain_id, domain_name, keywords = random.choice(self.DOMAINS)

        # Генерация названия
        name = self._generate_name(doc_type, domain_id, domain_name)

        # Генерация года и текста
        year = random.randint(2010, 2024)
        text = self._generate_text()

        # Определение требования
        is_requirement, requirement_score = self._is_requirement(doc_type, name, text)

        # Генерация тем
        num_themes = random.randint(1, 4)
        themes = random.sample(self.THEMES, num_themes)

        return {
            'id': f'doc_{doc_id}',
            'name': name,
            'type': doc_type,
            'type_label': doc_label,
            'domain': domain_id,
            'domain_name': domain_name,
            'year': year,
            'text': text,
            'is_requirement': int(is_requirement),
            'requirement_score': requirement_score,
            'themes': themes,
            'keywords': keywords,
            'length': len(text)
        }

    def _generate_name(self, doc_type, domain_id, domain_name):
        """Генерация названия документа"""
        if doc_type == 'federal_law':
            if domain_id == 'L02.1' and random.random() > 0.7:
                return "Федеральный закон №152-ФЗ 'О персональных данных'"
            number = random.randint(1, 500)
            return f"Федеральный закон №{number}-ФЗ"

        elif doc_type == 'government_decree':
            if domain_id == 'L02.1' and random.random() > 0.6:
                return "Постановление Правительства РФ №1119 'О требованиях к защите ПДн'"
            return f"Постановление Правительства РФ №{random.randint(1, 2000)}"

        elif doc_type == 'ministry_order':
            ministry = random.choice(self.MINISTRIES)
            return f"Приказ {ministry} №{random.randint(1, 1000)}"

        elif doc_type == 'fstek_document':
            return random.choice([
                "Базовая модель угроз безопасности ПДн",
                "Требования по безопасности информации",
                "Методика определения угроз",
                "Требования к СЗИ"
            ])

        elif doc_type == 'fsb_document':
            return "Требования ФСБ России к защите информации"

        elif doc_type == 'gost':
            return f"ГОСТ Р {random.randint(27000, 28000)}-2023"

        else:
            return f"Методические рекомендации по {domain_name.lower()}"

    def _generate_text(self):
        """Генерация текста документа"""
        text_length = random.randint(3, 8)
        return ". ".join(random.sample(self.TEXT_KEYWORDS, text_length)) + "."

    def _is_requirement(self, doc_type, name, text):
        """Определение, содержит ли требование к ИС"""
        requirement_score = 0.0

        # Веса для разных признаков
        if doc_type in ['federal_law', 'government_decree']:
            requirement_score += 0.7
        if 'требования' in name.lower():
            requirement_score += 0.3
        if any(word in text for word in ['требования к системе', 'технические средства', 'контроль доступа']):
            requirement_score += 0.2

        is_requirement = requirement_score > 0.8 or random.random() < requirement_score
        return is_requirement, requirement_score

    def _calculate_edge_weight(self, doc1, doc2):
        """расчет веса ребра"""
        weight = 0.0
        relationships = []

        # Связь по домену
        if doc1['domain'] == doc2['domain']:
            weight += 0.6
            relationships.append('same_domain')

        # Связь по типу
        if doc1['type'] == doc2['type']:
            weight += 0.5
            relationships.append('same_type')

        # Иерархическая связь
        if (doc1['type'] == 'federal_law' and doc2['type'] == 'government_decree') or \
           (doc2['type'] == 'federal_law' and doc1['type'] == 'government_decree'):
            weight += 0.8
            relationships.append('hierarchical')

        # Ссылочная связь
        common_keywords = set(doc1['keywords']).intersection(set(doc2['keywords']))
        if common_keywords:
            weight += 0.4 + 0.1 * len(common_keywords)
            relationships.append('referential')

        # Семантическая связь
        common_themes = set(doc1['themes']).intersection(set(doc2['themes']))
        if common_themes:
            weight += 0.3 + 0.15 * len(common_themes)
            relationships.append('semantic')

        if weight > 0:
            weight = min(1.0, weight / (len(relationships) or 1))
            if 'hierarchical' in relationships:
                weight = min(1.0, weight * 1.3)

        return relationships, weight

    def generate_dataset(self):
        """Генерация датасета"""
        print(f"Генерация {self.num_documents} документов...")

        # Генерация документов
        self.documents = [self.generate_document(i) for i in range(self.num_documents)]

        # Добавление узлов в граф
        for doc in self.documents:
            self.graph.add_node(doc['id'], **doc)

        # Генерация связей
        edges_added = 0
        for i in range(self.num_documents):
            doc1 = self.documents[i]
            # Выбираем случайные документы для связей
            num_connections = random.randint(2, 6)
            possible_indices = list(range(self.num_documents))
            possible_indices.remove(i)

            if len(possible_indices) > num_connections:
                targets = random.sample(possible_indices, num_connections)
            else:
                targets = possible_indices

            for j in targets:
                if i < j:  # Избегаем дублирования
                    doc2 = self.documents[j]
                    relationships, weight = self._calculate_edge_weight(doc1, doc2)

                    if weight > 0.3:  # Порог для создания связи
                        self.graph.add_edge(
                            doc1['id'], doc2['id'],
                            weight=weight,
                            relationships=relationships
                        )
                        edges_added += 1

        print(f"Создано {len(self.documents)} документов и {edges_added} связей")
        return pd.DataFrame(self.documents), self.graph

    def visualize_graph(self, figsize=(14, 10)):
        """визуализация графа"""
        plt.figure(figsize=figsize)

        # Цветовая схема
        colors = {
            'federal_law': '#FF6B6B',
            'government_decree': '#4ECDC4',
            'ministry_order': '#45B7D1',
            'fstek_document': '#96CEB4',
            'fsb_document': '#FFEAA7',
            'gost': '#DDA0DD',
            'methodology': '#98D8C8'
        }

        # Позиционирование
        pos = nx.spring_layout(self.graph, k=1.5, iterations=30, seed=42)

        # Подготовка данных для визуализации
        node_colors = [colors[self.graph.nodes[n]['type']] for n in self.graph.nodes()]
        node_sizes = [200 + 30 * self.graph.degree(n) for n in self.graph.nodes()]
        edge_widths = [2 + 5 * self.graph[u][v]['weight'] for u, v in self.graph.edges()]

        # Рисуем граф
        nx.draw_networkx_edges(self.graph, pos, alpha=0.4, width=edge_widths, edge_color='gray')
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes,
                              edgecolors='black', linewidths=1)

        # Упрощенные подписи
        labels = {n: f"{self.graph.nodes[n]['type_label'][:3]}\n{self.graph.nodes[n]['domain'][:5]}"
                 for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=7)

        plt.title(f'Граф НПА: {self.graph.number_of_nodes()} узлов, {self.graph.number_of_edges()} связей',
                 fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_statistics(self):
        """Быстрая статистика без сохранения файлов"""
        df = pd.DataFrame(self.documents)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Типы документов
        type_counts = df['type_label'].value_counts()
        axes[0, 0].bar(range(len(type_counts)), type_counts.values, color='skyblue')
        axes[0, 0].set_xticks(range(len(type_counts)))
        axes[0, 0].set_xticklabels(type_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Распределение по типам')
        axes[0, 0].set_ylabel('Количество')

        # 2. Требования
        req_counts = df['is_requirement'].value_counts()
        axes[0, 1].pie(req_counts.values, labels=['Не требование', 'Требование'],
                       autopct='%1.1f%%', colors=['lightgray', 'salmon'])
        axes[0, 1].set_title('Требования к ИС')

# 2.ПРОЦЕССОР

class NPADataProcessor:
    """процессор данных"""

    def __init__(self, graph, documents):
        self.graph = graph
        self.documents = documents
        self._cached_features = None
        self._cached_labels = None

    def _prepare_features_batch(self):
        """Пакетная подготовка признаков"""
        features_list = []
        node_to_idx = {}

        # Предварительные вычисления
        domains = sorted(set(d['domain'] for d in self.documents))
        doc_types = ['federal_law', 'government_decree', 'ministry_order',
                    'fstek_document', 'fsb_document', 'gost', 'methodology']
        key_terms = ['требования', 'безопасность', 'защита', 'система', 'контроль']

        for i, doc in enumerate(self.documents):
            node_to_idx[doc['id']] = i
            features = []

            # Тип документа
            features.extend([1 if doc['type'] == t else 0 for t in doc_types])

            # Домен
            features.extend([1 if doc['domain'] == d else 0 for d in domains])

            # Числовые признаки
            features.extend([
                (doc['year'] - 2010) / 14,  # Нормализованный год
                min(doc['length'] / 500, 1.0)  # Нормализованная длина
            ])

            # Ключевые слова
            text_lower = doc['text'].lower()
            features.extend([1 if term in text_lower else 0 for term in key_terms])

            # Темы
            features.append(len(doc['themes']) / 4)

            features_list.append(features)

        return torch.tensor(features_list, dtype=torch.float), node_to_idx

    def create_pyg_data(self, train_ratio=0.7, val_ratio=0.15):
        """Создание PyG данных с кэшированием"""
        # Кэширование признаков
        if self._cached_features is None:
            features, node_to_idx = self._prepare_features_batch()
            self._cached_features = features
        else:
            features = self._cached_features
            node_to_idx = {doc['id']: i for i, doc in enumerate(self.documents)}

        # Кэширование меток
        if self._cached_labels is None:
            self._cached_labels = torch.tensor([doc['is_requirement'] for doc in self.documents], dtype=torch.long)

        labels = self._cached_labels

        # Подготовка ребер
        edge_list = []
        edge_weights_list = []

        for u, v, data in self.graph.edges(data=True):
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            # Обе стороны для неориентированного графа
            edge_list.extend([[u_idx, v_idx], [v_idx, u_idx]])
            weight = data.get('weight', 0.5)
            edge_weights_list.extend([weight, weight])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights_list, dtype=torch.float)

        # Разделение данных
        n_nodes = len(self.documents)
        indices = np.arange(n_nodes)

        # Стратифицированное разделение
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio,
            stratify=labels.numpy(), random_state=42
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio/(1-train_ratio),
            stratify=labels[temp_idx].numpy(),
            random_state=42
        )

        # Создание масок
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        print(f"Разделение: Train={train_mask.sum().item()}, Val={val_mask.sum().item()}, Test={test_mask.sum().item()}")

        return Data(
            x=features, edge_index=edge_index, edge_attr=edge_weights,
            y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        ), node_to_idx

# 3. GNN МОДЕЛЬ

class GNNClassifier(nn.Module):
    """GNN модель"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_heads=4, dropout=0.3):
        super().__init__()

        # GAT слои с механизмом внимания
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)

        # Batch normalization и dropout
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Первый GAT слой
        x = self.conv1(x, edge_index)
        x = F.elu(self.bn1(x))
        x = self.dropout(x)

        # Второй GAT слой
        x = self.conv2(x, edge_index)
        x = F.elu(self.bn2(x))

        # Классификация
        return F.log_softmax(self.classifier(x), dim=1)

# 4. ЭКСПЕРИМЕНТ

class GNNExperiment:
    """эксперимент"""

    def __init__(self, data, model, device):
        self.data = data.to(device)
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': []
        }
        self.best_state = None
        self.best_val_f1 = 0

    def train_epoch(self, optimizer):
        self.model.train()
        optimizer.zero_grad()

        out = self.model(self.data)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accuracy на train
        pred = out.argmax(dim=1)
        train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).float().mean().item()

        return loss.item(), train_acc

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
            pred = out.argmax(dim=1)

            loss = F.nll_loss(out[mask], self.data.y[mask]).item()
            acc = (pred[mask] == self.data.y[mask]).float().mean().item()

            y_true = self.data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()

            f1 = f1_score(y_true, y_pred, zero_division=0)

            return {
                'loss': loss,
                'accuracy': acc,
                'f1': f1,
                'predictions': y_pred,
                'true_labels': y_true
            }

    def train(self, num_epochs=100, lr=0.005, weight_decay=5e-4, patience=20):
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        patience_counter = 0

        print("Начало обучения...")
        for epoch in range(1, num_epochs + 1):
            # Обучение
            train_loss, train_acc = self.train_epoch(optimizer)

            # Валидация
            val_results = self.evaluate(self.data.val_mask)

            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_acc'].append(val_results['accuracy'])
            self.history['val_f1'].append(val_results['f1'])

            # Обновление learning rate
            scheduler.step(val_results['f1'])

            # Сохранение лучшей модели
            if val_results['f1'] > self.best_val_f1:
                self.best_val_f1 = val_results['f1']
                self.best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Логирование
            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val F1: {val_results["f1"]:.4f}')

            # Ранняя остановка
            if patience_counter >= patience:
                print(f"\nРанняя остановка на эпохе {epoch}")
                break

        # Загрузка лучшей модели
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        print(f"\nОбучение завершено. Лучший Val F1: {self.best_val_f1:.4f}")
        return self.history

    def test(self):
        test_results = self.evaluate(self.data.test_mask)

        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("="*50)
        print(f"Accuracy:  {test_results['accuracy']:.4f}")
        print(f"F1-Score:  {test_results['f1']:.4f}")
        print(f"Loss:      {test_results['loss']:.4f}")

        # Краткий отчет
        print("\n" + "-"*50)
        print(classification_report(
            test_results['true_labels'],
            test_results['predictions'],
            target_names=['Не требование', 'Требование']
        ))

        return test_results

    def plot_training(self, figsize=(12, 4)):
        """визуализация обучения"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2, alpha=0.7)
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        # F1-Score
        axes[2].plot(epochs, self.history['val_f1'], 'g-', linewidth=2, alpha=0.7)
        axes[2].axhline(y=self.best_val_f1, color='r', linestyle='--',
                       label=f'Лучший: {self.best_val_f1:.3f}')
        axes[2].set_xlabel('Эпоха')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_title('Validation F1-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])

        plt.suptitle('История обучения GNN модели', fontsize=12, y=1.05)
        plt.tight_layout()
        plt.show()

# 5. ОПТИМИЗИРОВАННЫЙ АНАЛИЗАТОР

class ResultAnalyzer:
    """анализатор результатов"""

    @staticmethod
    def analyze_predictions(documents, predictions, node_to_idx):
        """Быстрый анализ предсказаний"""
        results = []
        for doc in documents:
            idx = node_to_idx[doc['id']]
            pred = predictions[idx]
            true = doc['is_requirement']

            results.append({
                'id': doc['id'][:10],
                'type': doc['type_label'],
                'domain': doc['domain_name'],
                'true': 'Требование' if true == 1 else 'Не требование',
                'pred': 'Требование' if pred == 1 else 'Не требование',
                'correct': pred == true
            })

        df = pd.DataFrame(results)

        print("\n" + "="*50)
        print("АНАЛИЗ ПРЕДСКАЗАНИЙ")
        print("="*50)

        # Общая точность
        accuracy = df['correct'].mean()
        print(f"\nОбщая точность: {accuracy:.2%}")

        # Точность по типам
        print("\nТочность по типам документов:")
        for doc_type in df['type'].unique():
            subset = df[df['type'] == doc_type]
            acc = subset['correct'].mean()
            print(f"  {doc_type}: {acc:.2%} ({len(subset)} документов)")

        # Матрица ошибок
        print("\nРаспределение предсказаний:")
        confusion = pd.crosstab(df['true'], df['pred'])
        print(confusion)

        return df

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Быстрая матрица ошибок"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Не требование', 'Требование'],
                   yticklabels=['Не требование', 'Требование'])
        plt.xlabel('Предсказано')
        plt.ylabel('Истинное значение')
        plt.title('Матрица ошибок')
        plt.tight_layout()
        plt.show()

# 6. ОПТИМИЗИРОВАННЫЙ ГЛАВНЫЙ ПРОЦЕСС

def run_optimized_pipeline(num_documents=50, epochs=100):
    """pipeline без создания файлов"""

    print("="*60)
    print("GNN ДЛЯ АНАЛИЗА НПА")
    print("="*60)

    # 1. Генерация данных
    print("\n[1/4] Генерация датасета...")
    generator = NPADatasetGenerator(num_documents)
    df, graph = generator.generate_dataset()

    # Быстрая статистика
    print(f"\nБаланс классов:")
    print(f"  Требования: {df['is_requirement'].sum()} ({df['is_requirement'].mean():.1%})")
    print(f"  Не требования: {len(df) - df['is_requirement'].sum()}")

    # Визуализация
    generator.visualize_graph(figsize=(12, 8))
    generator.show_statistics()

    # 2. Подготовка данных
    print("\n[2/4] Подготовка данных для GNN...")
    processor = NPADataProcessor(graph, generator.documents)
    data, node_to_idx = processor.create_pyg_data()

    print(f"\nХарактеристики графа:")
    print(f"  Узлы: {data.num_nodes}")
    print(f"  Ребра: {data.edge_index.shape[1] // 2}")
    print(f"  Признаки: {data.x.shape[1]}")

    # 3. Создание и обучение модели
    print("\n[3/4] Обучение модели...")
    model = GNNClassifier(
        input_dim=data.x.shape[1],
        hidden_dim=64,
        output_dim=2,
        num_heads=4,
        dropout=0.3
    )

    experiment = GNNExperiment(data, model, device)
    history = experiment.train(num_epochs=epochs)

    # Визуализация обучения
    experiment.plot_training()

    # 4. Тестирование и анализ
    print("\n[4/4] Тестирование и анализ...")
    test_results = experiment.test()

    # Матрица ошибок
    ResultAnalyzer.plot_confusion_matrix(
        test_results['true_labels'],
        test_results['predictions']
    )

    # Анализ предсказаний
    all_predictions = experiment.model(data).argmax(dim=1).cpu().numpy()
    df_analysis = ResultAnalyzer.analyze_predictions(
        generator.documents,
        all_predictions,
        node_to_idx
    )

    return experiment, df_analysis, test_results

if __name__ == "__main__":
    experiment, analysis_df, test_results = run_optimized_pipeline(
        num_documents=30,
        epochs=50
    )
