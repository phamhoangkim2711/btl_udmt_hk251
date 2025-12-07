import streamlit as st
import pandas as pd
import pulp
from typing import List, Dict, Any

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tối Ưu Hóa Bữa Trưa (Ràng buộc từng món)")

# Định nghĩa kiểu dữ liệu cho ràng buộc tùy chỉnh
CustomConstraint = Dict[str, Any]

# --- DỮ LIỆU VÀ HÀM CỐ ĐỊNH ---

# Dữ liệu mặc định ĐÃ THÊM CÁC CỘT ĐỊNH LƯỢNG CHO TỪNG MÓN
DEFAULT_FOODS_DATA = {
    'bread': {'cost': 5, 'cal_fat': 10, 'total_cal': 70, 'vit_c': 0, 'protein': 3, 'min_qty': 0, 'max_qty': 4},
    'peanut_butter': {'cost': 4, 'cal_fat': 75, 'total_cal': 100, 'vit_c': 0, 'protein': 4, 'min_qty': 0, 'max_qty': 2},
    'jelly': {'cost': 7, 'cal_fat': 0, 'total_cal': 50, 'vit_c': 3, 'protein': 0, 'min_qty': 0, 'max_qty': 2},
    'cracker': {'cost': 8, 'cal_fat': 20, 'total_cal': 60, 'vit_c': 0, 'protein': 1, 'min_qty': 0, 'max_qty': 5},
    'milk': {'cost': 15, 'cal_fat': 70, 'total_cal': 150, 'vit_c': 2, 'protein': 8, 'min_qty': 0, 'max_qty': 1},
    'juice': {'cost': 35, 'cal_fat': 0, 'total_cal': 100, 'vit_c': 120, 'protein': 1, 'min_qty': 0, 'max_qty': 1}
}

# DataFrame mặc định cho bảng ràng buộc Tổng Bữa Ăn
DEFAULT_AGGREGATE_CONSTRAINTS_DF = pd.DataFrame({
    'Nutrient': ['cost'],
    'Operator': ['<='],
    'Value': [200.0]
})

def run_optimization(foods_data: dict, aggregate_constraints: List[CustomConstraint]):
    """
    Hàm giải mô hình tối ưu hóa ăn trưa, có thêm ràng buộc min/max cho từng thực phẩm,
    và ràng buộc tổng cho cả bữa ăn.
    """
    if not foods_data:
        return None, "Lỗi: Không có dữ liệu thực phẩm để chạy mô hình."

    food_names = list(foods_data.keys())
    model = pulp.LpProblem("Lunch Optimization Flexible", pulp.LpMinimize)
    # x: Biến quyết định, là số lượng mỗi loại thực phẩm (số nguyên >= 0)
    x = pulp.LpVariable.dicts("X", food_names, lowBound=0, cat='Integer')

    # Lấy danh sách các thuộc tính hợp lệ từ thực phẩm đầu tiên
    valid_food_attributes = set(foods_data[food_names[0]].keys()) if food_names else set()

    # --- 1. HÀM MỤC TIÊU (Minimize Cost) ---
    if 'cost' not in valid_food_attributes:
        return None, "Lỗi: Dữ liệu thực phẩm phải có cột 'cost' để tối ưu hóa."

    model += (
        pulp.lpSum(foods_data[name]['cost'] * x[name] for name in food_names),
        "Total_Cost"
    )

    # --- 2. RÀNG BUỘC THEO TỪNG THỰC PHẨM (Item-Specific Constraints) ---
    # Sử dụng các cột 'min_qty' và 'max_qty' do người dùng nhập.
    
    if 'min_qty' in valid_food_attributes and 'max_qty' in valid_food_attributes:
        for name in food_names:
            min_val = foods_data[name].get('min_qty', 0)
            max_val = foods_data[name].get('max_qty', 100) # Giả định max mặc định là 100 nếu không được nhập

            # Ràng buộc Tối thiểu
            if min_val > 0:
                model += (x[name] >= min_val, f"Item_Min_Qty_{name}")
            
            # Ràng buộc Tối đa
            if max_val >= 0 and max_val < 100:
                model += (x[name] <= max_val, f"Item_Max_Qty_{name}")

    # --- 3. RÀNG BUỘC TỔNG BỮA ĂN CỐ ĐỊNH (Fixed Aggregate Constraints) ---
    
    # Ràng buộc Calo (C1, C2, C3)
    if 'total_cal' in valid_food_attributes and 'cal_fat' in valid_food_attributes:
        Total_Cal_Expr = pulp.lpSum(foods_data[name]['total_cal'] * x[name] for name in food_names)
        Cal_Fat_Expr = pulp.lpSum(foods_data[name]['cal_fat'] * x[name] for name in food_names)

        model += (Total_Cal_Expr >= 400, "Fixed_Min_Total_Calories")
        model += (Total_Cal_Expr <= 600, "Fixed_Max_Total_Calories")
        model += (Cal_Fat_Expr - 0.30 * Total_Cal_Expr <= 0, "Fixed_Max_30_Percent_Fat_Calories")

    # Ràng buộc Vitamin C (C4)
    if 'vit_c' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['vit_c'] * x[name] for name in food_names) >= 60, "Fixed_Min_Vitamin_C")

    # Ràng buộc Protein (C5)
    if 'protein' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['protein'] * x[name] for name in food_names) >= 12, "Fixed_Min_Protein")

    # RÀNG BUỘC ĐẶC BIỆT CỐ ĐỊNH KHÁC (C6, C7)
    # Lưu ý: Các ràng buộc này nên được thay bằng min_qty/max_qty ở mục 2
    # Nhưng giữ lại cho tính kế thừa của mô hình cũ.
    if 'bread' in food_names and 'min_qty' not in valid_food_attributes and 'max_qty' not in valid_food_attributes:
        # Nếu người dùng không nhập min/max, ta vẫn giữ ràng buộc cũ
        model += (x['bread'] == 2, "Fixed_Exact_2_Slices_Bread_Legacy")

    if 'peanut_butter' in food_names and 'jelly' in food_names:
        model += (x['peanut_butter'] - 2 * x['jelly'] >= 0, "Fixed_Peanut_Butter_vs_Jelly")

    # --- 4. RÀNG BUỘC TỔNG BỮA ĂN TÙY CHỈNH (Custom Aggregate Constraints) ---
    for i, constraint in enumerate(aggregate_constraints):
        nutrient = constraint.get('Nutrient', '').strip()
        operator = constraint.get('Operator', '').strip()
        value = constraint.get('Value', 0)

        # Kiểm tra tính hợp lệ và đảm bảo không trùng với các cột điều chỉnh số lượng min/max
        if nutrient in valid_food_attributes and nutrient not in ['min_qty', 'max_qty'] and operator in ['>=', '<=', '=']:
            total_expr = pulp.lpSum(foods_data[name].get(nutrient, 0) * x[name] for name in food_names)
            
            constraint_name = f"Custom_Aggregate_{i+1}_{nutrient}_{operator}_{value}"
            if operator == '>=':
                model += (total_expr >= value, constraint_name)
            elif operator == '<=':
                model += (total_expr <= value, constraint_name)
            elif operator == '=':
                model += (total_expr == value, constraint_name)

    # --- 5. GIẢI MÔ HÌNH ---
    model.solve()

    if model.status == pulp.LpStatusOptimal:
        optimal_cost = pulp.value(model.objective)
        results = {name: int(round(x[name].varValue)) 
                   for name in food_names 
                   if x[name].varValue is not None and x[name].varValue > 1e-6}
        return optimal_cost, results
    
    return None, pulp.LpStatus[model.status]


# --- HÀM CHÍNH CỦA STREAMLIT ---
def main():
    st.title("🥪 TỐI ƯU HÓA CHI PHÍ BỮA TRƯA (Ràng buộc từng món & Tổng bữa ăn)")
    st.markdown("Bạn có thể định nghĩa các chất dinh dưỡng, thêm **số lượng tối đa/tối thiểu** cho từng món, và đặt ràng buộc tổng cho cả bữa ăn.")
    
    st.divider()

    ## 1. PHẦN NHẬP DỮ LIỆU THỰC PHẨM
    st.header("1. Nhập và chỉnh sửa dữ liệu thực phẩm & Ràng buộc Số lượng")
    st.markdown("⚠️ **Lưu ý:**")
    st.markdown("* Cột **`cost`** là bắt buộc.")
    st.markdown("* Cột **`min_qty`** và **`max_qty`** cho phép bạn đặt ràng buộc số lượng riêng cho từng món ăn.")
    
    # --- CƠ CHẾ TÙY CHỈNH CỘT DỮ LIỆU ---
    
    if 'editable_df_v2' not in st.session_state:
        st.session_state.editable_df_v2 = pd.DataFrame.from_dict(DEFAULT_FOODS_DATA, orient='index')
        st.session_state.editable_df_v2.index.name = 'food_name'

    # Thiết lập cấu hình cột
    col_config = {}
    for col in st.session_state.editable_df_v2.columns:
        if col in ['min_qty', 'max_qty']:
            col_config[col] = st.column_config.NumberColumn(
                f"{col.replace('_', ' ').title()} (Ràng buộc)", 
                min_value=0,
                step=1,
                format="%d" # Chỉ cho phép số nguyên
            )
        elif col == 'cost':
             col_config[col] = st.column_config.NumberColumn(
                f"{col.replace('_', ' ').title()} (¢)", 
                min_value=0.0,
                format="%.2f",
                required=True
            )
        else:
            col_config[col] = st.column_config.NumberColumn(
                f"{col.replace('_', ' ').title()}", 
                min_value=0.0,
                format="%.2f"
            )

    edited_df = st.data_editor(
        st.session_state.editable_df_v2,
        column_config=col_config,
        num_rows="dynamic", 
        use_container_width=True,
        key="food_data_editor_v2"
    )
    
    st.session_state.editable_df_v2 = edited_df.copy()

    foods_input = edited_df.to_dict('index')
    
    if not edited_df.empty and len(edited_df.columns) > 0:
        valid_attributes = list(edited_df.columns)
    else:
        valid_attributes = []


    # --- KIỂM TRA LOGIC CƠ BẢN ---
    data_is_valid = True
    if 'cost' not in valid_attributes:
        st.error("❌ LỖI: Cột **`cost`** là bắt buộc để tối ưu hóa.")
        data_is_valid = False
    
    # Kiểm tra min_qty <= max_qty
    if 'min_qty' in valid_attributes and 'max_qty' in valid_attributes:
        for food_name, data in foods_input.items():
            if data.get('min_qty', 0) > data.get('max_qty', 100):
                st.error(
                    f"❌ LỖI LOGIC: Món **{food_name}** có Số lượng Tối thiểu ({data['min_qty']:.0f}) "
                    f"lớn hơn Số lượng Tối đa ({data['max_qty']:.0f})."
                )
                data_is_valid = False
                break

    st.divider()
    
    ## 2. RÀNG BUỘC TỔNG BỮA ĂN TÙY CHỈNH
    st.header("2. Thêm Ràng Buộc Tùy Chỉnh cho TỔNG BỮA ĂN")
    
    # Lọc danh sách thuộc tính hợp lệ cho ràng buộc tổng (loại bỏ min_qty, max_qty)
    aggregate_options = [attr for attr in valid_attributes if attr not in ['min_qty', 'max_qty']]
    st.markdown(f"**Các thuộc tính hợp lệ (đã nhập ở trên):** `{', '.join(aggregate_options)}`")

    custom_constraints_df = st.data_editor(
        DEFAULT_AGGREGATE_CONSTRAINTS_DF,
        column_config={
            "Nutrient": st.column_config.SelectboxColumn(
                "Chất dinh dưỡng",
                options=aggregate_options, # Tùy chỉnh danh sách
                required=True,
                help="Chọn thuộc tính tổng của cả bữa ăn."
            ),
            "Operator": st.column_config.SelectboxColumn(
                "Toán tử",
                options=['>=', '<=', '='],
                required=True,
                help="Chọn toán tử so sánh (>=, <=, =)."
            ),
            "Value": st.column_config.NumberColumn(
                "Giá trị",
                min_value=0.0,
                format="%.2f",
                required=True,
                help="Nhập giá trị mục tiêu của ràng buộc."
            )
        },
        num_rows="dynamic",
        use_container_width=True,
        key="custom_constraints_editor_v2"
    )
    
    aggregate_constraints = custom_constraints_df.to_dict('records')

    st.divider()

    ## 3. PHẦN CHẠY MÔ HÌNH VÀ KẾT QUẢ
    st.header("3. Kết quả tối ưu hóa")

    if st.button("🚀 Chạy mô hình tối ưu", disabled=not data_is_valid or edited_df.empty):
        
        # Chạy mô hình PuLP
        optimal_cost, result_data = run_optimization(foods_input, aggregate_constraints)

        if optimal_cost is not None and isinstance(result_data, dict):
            st.success("✅ **ĐÃ TÌM THẤY KẾT QUẢ TỐI ƯU**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Chi phí tối thiểu", f"{optimal_cost:.2f} ¢")
            
            solution_df = pd.DataFrame(
                result_data.items(), 
                columns=['Thực phẩm', 'Số lượng tối ưu']
            )
            solution_df['Số lượng tối ưu'] = solution_df['Số lượng tối ưu'].astype(int)
            
            with col2:
                st.dataframe(solution_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")

            # --- KIỂM TRA CÁC RÀNG BUỘC SAU KHI TỐI ƯU ---
            st.subheader("Kiểm tra Giá trị đạt được:")
            
            # Tính toán tất cả các thuộc tính của giải pháp tối ưu
            display_attributes = ['cost'] + sorted([attr for attr in aggregate_options if attr != 'cost'])
            
            summary_data = {'Chỉ Số': [], 'Giá Trị Đạt Được (Tổng)': [], 'Ràng Buộc Mục Tiêu/Cố Định': []}
            
            # 1. Tính toán giá trị đạt được cho các thuộc tính hợp lệ
            for attr in display_attributes:
                current_value = sum(foods_input[name].get(attr, 0) * result_data.get(name, 0) for name in foods_input)
                
                # --- Xác định Ràng Buộc Cố Định (Fixed Aggregate Constraint) ---
                fixed_req = 'N/A'
                if attr == 'cost': fixed_req = 'Minimize'
                elif attr == 'total_cal': fixed_req = '400 <= Value <= 600'
                elif attr == 'cal_fat': 
                    total_cal_value = sum(foods_input[name].get('total_cal', 0) * result_data.get(name, 0) for name in foods_input)
                    fixed_req = f'<= 30% Tổng Calo ({0.30 * total_cal_value:.2f})' if 'total_cal' in valid_attributes else 'N/A'
                elif attr == 'vit_c': fixed_req = '>= 60 mg'
                elif attr == 'protein': fixed_req = '>= 12 g'
                
                summary_data['Chỉ Số'].append(attr.replace('_', ' ').title())
                summary_data['Giá Trị Đạt Được (Tổng)'].append(f"{current_value:.2f}")
                summary_data['Ràng Buộc Mục Tiêu/Cố Định'].append(fixed_req)

            # 2. Thêm các ràng buộc tùy chỉnh (Aggregate)
            for i, constraint in enumerate(aggregate_constraints):
                 nutrient = constraint.get('Nutrient', '').strip()
                 operator = constraint.get('Operator', '').strip()
                 value = constraint.get('Value', 0)
                 
                 if nutrient and operator in ['>=', '<=', '='] and nutrient in aggregate_options:
                     achieved_value = sum(foods_input[name].get(nutrient, 0) * result_data.get(name, 0) for name in foods_input)

                     summary_data['Chỉ Số'].append(f"Custom: {nutrient.replace('_', ' ').title()}")
                     summary_data['Giá Trị Đạt Được (Tổng)'].append(f"{achieved_value:.2f}")
                     summary_data['Ràng Buộc Mục Tiêu/Cố Định'].append(f"{operator} {value:.2f}")
            
            st.table(pd.DataFrame(summary_data))
            
        else:
            st.error(f"❌ KHÔNG TÌM THẤY LỜI GIẢI TỐI ƯU. **Trạng thái**: {result_data}. Vui lòng kiểm tra lại các ràng buộc hoặc dữ liệu nhập.")
        
    st.caption("Mô hình được giải quyết bằng PuLP (Integer Linear Programming).")

if __name__ == "__main__":
    main()
