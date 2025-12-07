import streamlit as st
import pandas as pd
import pulp
from typing import List, Dict, Any

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tối Ưu Hóa Bữa Trưa (Ràng buộc từng món)")

# Định nghĩa kiểu dữ liệu cho ràng buộc tùy chỉnh
CustomConstraint = Dict[str, Any]

# --- DỮ LIỆU MẶC ĐỊNH ---

# Dữ liệu mặc định ĐÃ THÊM CÁC CỘT ĐỊNH LƯỢNG CHO TỪNG MÓN
DEFAULT_FOODS_DATA = {
    'bread': {'cost': 5.0, 'cal_fat': 10.0, 'total_cal': 70.0, 'vit_c': 0.0, 'protein': 3.0, 'min_qty': 0, 'max_qty': 4},
    'peanut_butter': {'cost': 4.0, 'cal_fat': 75.0, 'total_cal': 100.0, 'vit_c': 0.0, 'protein': 4.0, 'min_qty': 0, 'max_qty': 2},
    'jelly': {'cost': 7.0, 'cal_fat': 0.0, 'total_cal': 50.0, 'vit_c': 3.0, 'protein': 0.0, 'min_qty': 0, 'max_qty': 2},
    'cracker': {'cost': 8.0, 'cal_fat': 20.0, 'total_cal': 60.0, 'vit_c': 0.0, 'protein': 1.0, 'min_qty': 0, 'max_qty': 5},
    'milk': {'cost': 15.0, 'cal_fat': 70.0, 'total_cal': 150.0, 'vit_c': 2.0, 'protein': 8.0, 'min_qty': 0, 'max_qty': 1},
    'juice': {'cost': 35.0, 'cal_fat': 0.0, 'total_cal': 100.0, 'vit_c': 120.0, 'protein': 1.0, 'min_qty': 0, 'max_qty': 1}
}

# DataFrame mặc định cho bảng ràng buộc Tổng Bữa Ăn
DEFAULT_AGGREGATE_CONSTRAINTS_DF = pd.DataFrame({
    'Nutrient': ['cost'],
    'Operator': ['<='],
    'Value': [200.0]
})

# Khởi tạo trạng thái phiên (Session State)
def initialize_session_state():
    """Khởi tạo tất cả các DataFrame trong session state nếu chưa có."""
    if 'food_df' not in st.session_state:
        st.session_state.food_df = pd.DataFrame.from_dict(DEFAULT_FOODS_DATA, orient='index')
        st.session_state.food_df.index.name = 'food_name'
    if 'constraints_df' not in st.session_state:
        st.session_state.constraints_df = DEFAULT_AGGREGATE_CONSTRAINTS_DF.copy()

# --- HÀM TỐI ƯU HÓA PU.L.P ---

def run_optimization(foods_data: dict, aggregate_constraints: List[CustomConstraint]):
    """
    Hàm giải mô hình tối ưu hóa ăn trưa.
    Trả về: (optimal_cost, results) hoặc (None, status_message)
    """
    if not foods_data:
        return None, "Lỗi: Không có dữ liệu thực phẩm để chạy mô hình."

    food_names = list(foods_data.keys())
    model = pulp.LpProblem("Lunch Optimization Flexible", pulp.LpMinimize)
    # x: Biến quyết định, là số lượng mỗi loại thực phẩm (số nguyên >= 0)
    x = pulp.LpVariable.dicts("X", food_names, lowBound=0, cat='Integer')

    # Lấy danh sách các thuộc tính hợp lệ
    valid_food_attributes = set(foods_data[food_names[0]].keys()) if food_names else set()

    # --- 1. HÀM MỤC TIÊU (Minimize Cost) ---
    if 'cost' not in valid_food_attributes:
        return None, "Lỗi: Dữ liệu thực phẩm phải có cột 'cost' để tối ưu hóa."

    model += (
        pulp.lpSum(foods_data[name]['cost'] * x[name] for name in food_names),
        "Total_Cost"
    )

    # --- 2. RÀNG BUỘC THEO TỪNG THỰC PHẨM (Item-Specific Constraints) ---
    if 'min_qty' in valid_food_attributes and 'max_qty' in valid_food_attributes:
        for name in food_names:
            min_val = foods_data[name].get('min_qty', 0)
            max_val = foods_data[name].get('max_qty', float('inf')) 

            # Ràng buộc Tối thiểu
            if min_val > 0:
                model += (x[name] >= min_val, f"Item_Min_Qty_{name}")
            
            # Ràng buộc Tối đa (Chỉ thêm nếu có giới hạn cụ thể)
            if max_val >= 0 and max_val != float('inf'):
                model += (x[name] <= max_val, f"Item_Max_Qty_{name}")

    # --- 3. RÀNG BUỘC TỔNG BỮA ĂN CỐ ĐỊNH (Fixed Aggregate Constraints) ---
    
    # Ràng buộc Calo (C1, C2)
    if 'total_cal' in valid_food_attributes:
        Total_Cal_Expr = pulp.lpSum(foods_data[name]['total_cal'] * x[name] for name in food_names)
        model += (Total_Cal_Expr >= 400, "Fixed_Min_Total_Calories")
        model += (Total_Cal_Expr <= 600, "Fixed_Max_Total_Calories")

    # Ràng buộc Chất béo (C3) - dựa trên Calo tổng
    if 'cal_fat' in valid_food_attributes and 'total_cal' in valid_food_attributes:
        Cal_Fat_Expr = pulp.lpSum(foods_data[name]['cal_fat'] * x[name] for name in food_names)
        model += (Cal_Fat_Expr - 0.30 * Total_Cal_Expr <= 0, "Fixed_Max_30_Percent_Fat_Calories")

    # Ràng buộc Vitamin C (C4)
    if 'vit_c' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['vit_c'] * x[name] for name in food_names) >= 60, "Fixed_Min_Vitamin_C")

    # Ràng buộc Protein (C5)
    if 'protein' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['protein'] * x[name] for name in food_names) >= 12, "Fixed_Min_Protein")

    # RÀNG BUỘC ĐẶC BIỆT CỐ ĐỊNH (C6, C7) - Chỉ giữ lại cho tính kế thừa
    if 'peanut_butter' in food_names and 'jelly' in food_names:
        model += (x['peanut_butter'] - 2 * x['jelly'] >= 0, "Fixed_Peanut_Butter_vs_Jelly")

    # --- 4. RÀNG BUỘC TỔNG BỮA ĂN TÙY CHỈNH (Custom Aggregate Constraints) ---
    for i, constraint in enumerate(aggregate_constraints):
        nutrient = str(constraint.get('Nutrient', '')).strip()
        operator = str(constraint.get('Operator', '')).strip()
        value = constraint.get('Value', 0)

        # Kiểm tra tính hợp lệ
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
    try:
        model.solve()
    except Exception as e:
        return None, f"Lỗi trong quá trình giải mô hình: {e}"

    if model.status == pulp.LpStatusOptimal:
        optimal_cost = pulp.value(model.objective)
        results = {name: int(round(x[name].varValue)) 
                   for name in food_names 
                   if x[name].varValue is not None and x[name].varValue > 1e-6}
        return optimal_cost, results
    
    return None, pulp.LpStatus[model.status]


# --- HÀM CHÍNH CỦA STREAMLIT ---
def main():
    initialize_session_state()
    
    st.title("🥪 TỐI ƯU HÓA CHI PHÍ BỮA TRƯA")
    st.markdown("Định nghĩa thực phẩm, ràng buộc số lượng từng món, và đặt ràng buộc tổng cho bữa ăn.")
    
    st.divider()

    ## 1. PHẦN NHẬP DỮ LIỆU THỰC PHẨM & RÀNG BUỘC SỐ LƯỢNG
    st.header("1. Nhập và chỉnh sửa dữ liệu thực phẩm & Ràng buộc Số lượng")
    st.markdown("* Cột **`cost`** là bắt buộc. Cột **`min_qty`** và **`max_qty`** áp dụng ràng buộc số lượng riêng cho từng món.")
    
    # Thiết lập cấu hình cột
    col_config = {
        'food_name': st.column_config.TextColumn("Tên Món", required=True),
        'cost': st.column_config.NumberColumn("Cost (¢)", min_value=0.0, format="%.2f", required=True),
        'min_qty': st.column_config.NumberColumn("Min Qty (Ràng buộc)", min_value=0, step=1, format="%d"),
        'max_qty': st.column_config.NumberColumn("Max Qty (Ràng buộc)", min_value=0, step=1, format="%d"),
        'cal_fat': st.column_config.NumberColumn("Cal Fat", min_value=0.0, format="%.2f"),
        'total_cal': st.column_config.NumberColumn("Total Cal", min_value=0.0, format="%.2f"),
        'vit_c': st.column_config.NumberColumn("Vit C", min_value=0.0, format="%.2f"),
        'protein': st.column_config.NumberColumn("Protein", min_value=0.0, format="%.2f"),
    }
    
    # Lấy DataFrame từ session state và chỉnh sửa
    edited_df = st.data_editor(
        st.session_state.food_df,
        column_config=col_config,
        num_rows="dynamic", 
        use_container_width=True,
        key="food_data_editor_v3"
    )
    
    # Cập nhật Session State ngay lập tức
    st.session_state.food_df = edited_df.copy()

    foods_input = edited_df.to_dict('index')
    
    # Lấy danh sách thuộc tính hiện tại
    if not edited_df.empty:
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
            min_val = data.get('min_qty', 0)
            max_val = data.get('max_qty', 9999) # Dùng giá trị lớn cho None/NaN

            if pd.isna(min_val): min_val = 0
            if pd.isna(max_val): max_val = 9999

            if min_val > max_val:
                st.error(
                    f"❌ LỖI LOGIC: Món **{food_name}** có Min Qty ({min_val:.0f}) "
                    f"lớn hơn Max Qty ({max_val:.0f})."
                )
                data_is_valid = False
                break
    
    st.divider()
    
    ## 2. RÀNG BUỘC TỔNG BỮA ĂN TÙY CHỈNH
    st.header("2. Thêm Ràng Buộc Tùy Chỉnh cho TỔNG BỮA ĂN")
    
    # Lọc danh sách thuộc tính hợp lệ cho ràng buộc tổng (loại bỏ min_qty, max_qty)
    aggregate_options = [attr for attr in valid_attributes if attr not in ['min_qty', 'max_qty']]
    st.markdown(f"**Các thuộc tính hợp lệ:** `{', '.join(aggregate_options)}`")

    # Lấy DataFrame ràng buộc từ session state
    custom_constraints_df = st.data_editor(
        st.session_state.constraints_df,
        column_config={
            "Nutrient": st.column_config.SelectboxColumn(
                "Chất dinh dưỡng", options=aggregate_options, required=True
            ),
            "Operator": st.column_config.SelectboxColumn(
                "Toán tử", options=['>=', '<=', '='], required=True
            ),
            "Value": st.column_config.NumberColumn(
                "Giá trị", min_value=0.0, format="%.2f", required=True
            )
        },
        num_rows="dynamic",
        use_container_width=True,
        key="custom_constraints_editor_v3"
    )
    
    # Cập nhật Session State
    st.session_state.constraints_df = custom_constraints_df.copy()
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
                elif attr == 'cal_fat' and 'total_cal' in valid_attributes:
                    # Tính lại total_cal của giải pháp
                    total_cal_value = sum(foods_input[name].get('total_cal', 0) * result_data.get(name, 0) for name in foods_input)
                    fixed_req = f'<= 30% Tổng Calo ({0.30 * total_cal_value:.2f})'
                elif attr == 'vit_c': fixed_req = '>= 60 g'
                elif attr == 'protein': fixed_req = '>= 12 g'
                
                # Thêm ràng buộc tùy chỉnh vào mô tả nếu nó là cost, hoặc chưa có ràng buộc cố định
                for constraint in aggregate_constraints:
                    if constraint.get('Nutrient') == attr:
                         # Nếu đã có ràng buộc cố định, ta có thể ghi đè hoặc thêm
                         if fixed_req == 'N/A' or attr == 'cost':
                             fixed_req = f"{constraint['Operator']} {constraint['Value']:.2f}"
                             break
                         else: # Nếu đã có cố định, thêm ràng buộc tùy chỉnh vào mô tả
                             fixed_req += f"; Custom: {constraint['Operator']} {constraint['Value']:.2f}"
                             break


                summary_data['Chỉ Số'].append(attr.replace('_', ' ').title())
                summary_data['Giá Trị Đạt Được (Tổng)'].append(f"{current_value:.2f}")
                summary_data['Ràng Buộc Mục Tiêu/Cố Định'].append(fixed_req)

            st.table(pd.DataFrame(summary_data))
            
        else:
            status_msg = result_data if isinstance(result_data, str) else "Lỗi không xác định"
            st.error(f"❌ KHÔNG TÌM THẤY LỜI GIẢI TỐI ƯU. **Trạng thái**: {status_msg}. Vui lòng kiểm tra lại các ràng buộc hoặc dữ liệu nhập.")
        
    st.caption("Mô hình được giải quyết bằng PuLP (Integer Linear Programming).")

if __name__ == "__main__":
    main()
