from collections import defaultdict
from itertools import combinations
from data_clean import data_translation


class Eclat:
    def __init__(self, min_support=0.1):
        self.min_support = min_support
        self.transactions = []
        self.items_vertical = defaultdict(set)
        self.frequent_itemsets = []

    def fit(self, transactions):
        """训练Eclat模型"""
        self.transactions = transactions
        self.n_transactions = len(transactions)

        # 构建垂直数据格式（项集-事务ID）
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                self.items_vertical[item].add(tid)

        # 获取满足最小支持度的单项集
        self.frequent_items = {item: tids for item, tids in self.items_vertical.items()
                               if len(tids) / self.n_transactions >= self.min_support}

        # 寻找频繁项集
        self._eclat([], sorted(self.frequent_items.keys()))

        return self

    def _eclat(self, prefix, items):
        """Eclat算法"""
        for i in range(len(items)):
            item = items[i]
            item_tids = self.items_vertical[item]

            # 当前项集
            current_itemset = prefix + [item]
            support = len(item_tids) / self.n_transactions

            if support >= self.min_support:
                self.frequent_itemsets.append((current_itemset, support))

                # 获取后续可能的项集
                suffix_items = items[i + 1:]
                if suffix_items:
                    new_items = []
                    for next_item in suffix_items:
                        next_tids = self.items_vertical[next_item]
                        intersection_tids = item_tids & next_tids

                        if len(intersection_tids) / self.n_transactions >= self.min_support:
                            new_items.append(next_item)
                            # 更新交集
                            self.items_vertical[tuple(
                                current_itemset + [next_item])] = intersection_tids

                    if new_items:
                        self._eclat(current_itemset, new_items)

    def generate_rules(self, min_confidence=0.5):
        """生成关联规则"""
        rules = []
        for itemset, support in self.frequent_itemsets:
            if len(itemset) < 2:
                continue

            # 生成所有可能的规则
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = list(antecedent)
                    consequent = list(set(itemset) - set(antecedent))

                    # 计算置信度
                    ant_support = len(
                        self.items_vertical[antecedent[0]]) / self.n_transactions
                    confidence = support / ant_support

                    if confidence >= min_confidence:
                        lift = confidence / \
                            (len(
                                self.items_vertical[consequent[0]]) / self.n_transactions)
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift
                        })

        return sorted(rules, key=lambda x: x['lift'], reverse=True)


# 分析商品
def analyze_shopping_data(transactions, min_support=0.1, min_confidence=0.5):

    eclat = Eclat(min_support=min_support)
    eclat.fit(transactions)

    # 生成关联规则
    rules = eclat.generate_rules(min_confidence=min_confidence)

    print(f"\n发现的频繁项集数量: {len(eclat.frequent_itemsets)}")
    print("\n===== top 5 关联规则 =====")

    for i, rule in enumerate(rules[:5], 1):
        print(f"\n规则 {i}:")
        print(f"如果购买: {' + '.join(rule['antecedent'])}")
        print(f"则可能购买: {' + '.join(rule['consequent'])}")
        print(f"支持度: {rule['support']:.3f}")
        print(f"置信度: {rule['confidence']:.3f}")
        print(f"提升度: {rule['lift']:.3f}")

    return eclat, rules


if __name__ == "__main__":
    data_list = data_translation()

    eclat_model, association_rules = analyze_shopping_data(
        data_list,
        min_support=0.05,
        min_confidence=0.3
    )
