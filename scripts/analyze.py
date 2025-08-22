import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

class KernelAnalyzer:
    def __init__(self, csv_file="times.csv"):
        self.csv_file = csv_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load dữ liệu từ CSV"""
        try:
            self.df = pd.read_csv(self.csv_file)
            # Xoá các dòng có giá trị time_ns không hợp lệ (không phải string số)
            self.df = self.df.dropna(subset=['name', 'time_ms'])
            self.df['time_ms'] = pd.to_numeric(self.df['time_ms'], errors='coerce')
            print(f"📊 Loaded {len(self.df)} kernel executions")
        except FileNotFoundError:
            print(f"❌ File {self.csv_file} not found!")
            return
    
    def basic_stats(self):
        """Thống kê cơ bản"""
        print("\n" + "="*50)
        print("📈 BASIC STATISTICS")
        print("="*50)
        
        stats = self.df.groupby('name')['time_ms'].agg([
            'count', 'mean', 'std', 'min', 'max', 'sum'
        ]).round(4)
        
        # Sort by total time
        stats = stats.sort_values('sum', ascending=False)
        print(stats)
        
        return stats
    
    def performance_breakdown(self):
        """Phân tích performance"""
        print("\n" + "="*50)
        print("⚡ PERFORMANCE BREAKDOWN")
        print("="*50)
        
        total_time = self.df['time_ms'].sum()
        kernel_times = self.df.groupby('name')['time_ms'].sum().sort_values(ascending=False)
        
        print(f"Total execution time: {total_time:.3f} ms\n")
        
        print("Time distribution:")
        for kernel, time in kernel_times.items():
            percentage = (time / total_time) * 100
            print(f"  {kernel:<25} {time:>8.3f} ms ({percentage:>5.1f}%)")
    
    def find_bottlenecks(self):
        """Tìm bottleneck"""
        print("\n" + "="*50)
        print("🔍 BOTTLENECK ANALYSIS")
        print("="*50)
        
        kernel_times = self.df.groupby('name')['time_ms'].sum().sort_values(ascending=False)
        total_time = self.df['time_ms'].sum()
        
        # Top 3 slowest
        top3 = kernel_times.head(3)
        print("🐌 Top 3 slowest kernels:")
        for i, (kernel, time) in enumerate(top3.items(), 1):
            percentage = (time / total_time) * 100
            print(f"  {i}. {kernel}: {time:.3f} ms ({percentage:.1f}%)")
        
        # Kernels taking > 10% of total time
        bottlenecks = kernel_times[kernel_times / total_time > 0.1]
        if len(bottlenecks) > 0:
            print(f"\n⚠️  {len(bottlenecks)} kernel(s) taking >10% of total time:")
            for kernel, time in bottlenecks.items():
                percentage = (time / total_time) * 100
                print(f"    {kernel}: {percentage:.1f}%")
    
    def variability_analysis(self):
        """Phân tích độ biến thiên"""
        print("\n" + "="*50)
        print("📊 VARIABILITY ANALYSIS")
        print("="*50)
        
        # Chỉ analyze kernels được gọi > 1 lần
        multi_runs = self.df.groupby('name').filter(lambda x: len(x) > 1)
        
        if len(multi_runs) == 0:
            print("No kernels with multiple executions found.")
            return
        
        variability = multi_runs.groupby('name')['time_ms'].agg(['mean', 'std']).round(4)
        variability['cv'] = (variability['std'] / variability['mean'] * 100).round(2)
        variability = variability.sort_values('cv', ascending=False)
        
        print("Coefficient of Variation (CV = std/mean * 100%):")
        for kernel, row in variability.iterrows():
            if row['cv'] > 10:
                flag = "⚠️ "
            else:
                flag = "✅ "
            print(f"  {flag}{kernel}: {row['cv']}% (mean: {row['mean']:.3f}ms)")
    
    def execution_pattern(self):
        """Phân tích pattern thực thi"""
        print("\n" + "="*50)
        print("🔄 EXECUTION PATTERN")
        print("="*50)
        
        execution_counts = self.df['name'].value_counts().sort_values(ascending=False)
        
        print("Execution counts:")
        for kernel, count in execution_counts.items():
            print(f"  {kernel:<25} {count:>3} times")
        
        # Kernels chỉ chạy 1 lần
        single_runs = execution_counts[execution_counts == 1]
        if len(single_runs) > 0:
            print(f"\n📝 {len(single_runs)} kernel(s) executed only once:")
            for kernel in single_runs.index:
                time = self.df[self.df['name'] == kernel]['time_ms'].iloc[0]
                print(f"    {kernel}: {time:.3f} ms")
    
    def create_visualizations(self):
        """Tạo charts"""
        if len(self.df) == 0:
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Time distribution pie chart
        kernel_times = self.df.groupby('name')['time_ms'].sum()
        top_kernels = kernel_times.nlargest(8)
        others = kernel_times.sum() - top_kernels.sum()
        
        if others > 0:
            plot_data = pd.concat([top_kernels, pd.Series([others], index=['Others'])])
        else:
            plot_data = top_kernels
        
        axes[0,0].pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Execution Time Distribution')
        
        # 2. Bar chart - total time per kernel
        kernel_times_sorted = kernel_times.sort_values(ascending=True)
        axes[0,1].barh(range(len(kernel_times_sorted)), kernel_times_sorted.values)
        axes[0,1].set_yticks(range(len(kernel_times_sorted)))
        axes[0,1].set_yticklabels([name[:20] for name in kernel_times_sorted.index])
        axes[0,1].set_xlabel('Time (ms)')
        axes[0,1].set_title('Total Time per Kernel')
        
        # 3. Execution sequence
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.df['name'].unique())))
        kernel_colors = dict(zip(self.df['name'].unique(), colors))
        
        for i, (_, row) in enumerate(self.df.iterrows()):
            axes[1,0].bar(i, row['time_ms'], color=kernel_colors[row['name']], alpha=0.7)
        axes[1,0].set_xlabel('Execution Order')
        axes[1,0].set_ylabel('Time (ms)')
        axes[1,0].set_title('Execution Timeline')
        
        # 4. Box plot for kernels with multiple runs
        multi_runs = self.df.groupby('name').filter(lambda x: len(x) > 1)
        if len(multi_runs) > 0:
            multi_runs.boxplot(column='time_ms', by='name', ax=axes[1,1])
            axes[1,1].set_title('Time Distribution (Multi-run Kernels)')
            axes[1,1].set_xlabel('Kernel')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'No kernels with\nmultiple executions', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Time Distribution')
        
        plt.tight_layout()
        plt.savefig('kernel_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Charts saved as 'kernel_analysis.png'")
        plt.show()
    
    def optimization_suggestions(self):
        """Đề xuất tối ưu"""
        print("\n" + "="*50)
        print("💡 OPTIMIZATION SUGGESTIONS")
        print("="*50)
        
        total_time = self.df['time_ms'].sum()
        kernel_times = self.df.groupby('name')['time_ms'].sum().sort_values(ascending=False)
        
        # Kernels chiếm > 20% thời gian
        major_bottlenecks = kernel_times[kernel_times / total_time > 0.2]
        if len(major_bottlenecks) > 0:
            print("🎯 Priority 1 - Major bottlenecks (>20% of total time):")
            for kernel, time in major_bottlenecks.items():
                percentage = (time / total_time) * 100
                print(f"    Optimize {kernel} ({percentage:.1f}% of total time)")
        
        # Kernels chiếm 5-20% thời gian
        medium_bottlenecks = kernel_times[(kernel_times / total_time > 0.05) & 
                                        (kernel_times / total_time <= 0.2)]
        if len(medium_bottlenecks) > 0:
            print("\n🔧 Priority 2 - Medium impact (5-20% of total time):")
            for kernel, time in medium_bottlenecks.items():
                percentage = (time / total_time) * 100
                print(f"    Consider optimizing {kernel} ({percentage:.1f}%)")
        
        # Memory transfer analysis
        memory_kernels = self.df[self.df['name'].str.contains('copy|memcpy', case=False)]
        if len(memory_kernels) > 0:
            memory_time = memory_kernels['time_ms'].sum()
            memory_percentage = (memory_time / total_time) * 100
            print(f"\n💾 Memory operations: {memory_time:.3f} ms ({memory_percentage:.1f}%)")
            if memory_percentage > 10:
                print("    Consider reducing memory transfers or using async copies")
    
    def export_report(self, filename="analysis_report.txt"):
        """Export báo cáo"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("KERNEL PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic stats
            total_time = self.df['time_ms'].sum()
            f.write(f"Total execution time: {total_time:.3f} ms\n")
            f.write(f"Number of kernel calls: {len(self.df)}\n")
            f.write(f"Unique kernels: {self.df['name'].nunique()}\n\n")
            
            # Top kernels
            kernel_times = self.df.groupby('name')['time_ms'].sum().sort_values(ascending=False)
            f.write("Top 5 most time-consuming kernels:\n")
            for i, (kernel, time) in enumerate(kernel_times.head(5).items(), 1):
                percentage = (time / total_time) * 100
                f.write(f"{i}. {kernel}: {time:.3f} ms ({percentage:.1f}%)\n")
        
        print(f"📄 Report exported to '{filename}'")
    
    def run_analysis(self):
        """Chạy toàn bộ phân tích"""
        if self.df is None:
            return
        
        print("🚀 Starting Kernel Performance Analysis...")
        
        self.basic_stats()
        self.performance_breakdown()
        self.find_bottlenecks()
        self.variability_analysis()
        self.execution_pattern()
        self.optimization_suggestions()
        self.create_visualizations()
        self.export_report()
        
        print("\n✅ Analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Analyze kernel performance')
    parser.add_argument('--file', '-f', default='times.csv', help='CSV file path')
    args = parser.parse_args()
    
    analyzer = KernelAnalyzer(args.file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()